from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_ABSOLUTE_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .dual_coordinate_melody import (
    INPUT_NOTE_GROUP,
    INPUT_REST,
    PITCH_DELTA_OFFSET,
    PITCH_RELATION_CLASSES,
    PITCH_RELATION_NEUTRAL,
    TIME_RELATION_BUCKETS,
    FactorizedNextUnitDecoder,
    MelodyCausalBatch,
    signed_log_bucket,
)
from .tonal_frame_factorization_melody import (
    TonalFrameFactorizedCore,
    tonic_so2_coordinates,
)
from .tonic_coordinate_melody import TonicCoordinateBatch


M4L_PHASE4B_STEP1_SCHEMA: Final = (
    "usmm.m4l.phase4b.musical_time_representation.step1.v1"
)
PHASE4B_STEP1_BASELINE: Final = "S0_PHASE4A1_B2"
PHASE4B_STEP1_MUSICAL_TIME: Final = "S1_MUSICAL_TIME_GEOMETRY"
PHASE4B_STEP1_ARMS: Final = (
    PHASE4B_STEP1_BASELINE,
    PHASE4B_STEP1_MUSICAL_TIME,
)


def cyclic_phase(
    values: torch.Tensor, period: int, *, dtype: torch.dtype
) -> torch.Tensor:
    """Encode a known physical cycle without claiming semantic periodicity."""

    if int(period) <= 0:
        raise ValueError("cyclic period must be positive")
    angle = values.to(dtype) * (2.0 * math.pi / float(period))
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def signed_unit_distance(values: torch.Tensor, unit: int) -> torch.Tensor:
    """Return symmetric signed whole-unit distance without negative flooring."""

    if int(unit) <= 0:
        raise ValueError("distance unit must be positive")
    magnitude = torch.div(values.abs(), int(unit), rounding_mode="floor")
    return magnitude * values.sign()


class MusicalTimeFactorizedCore(TonalFrameFactorizedCore):
    """Phase 4A.1 B2 with a corrected, explicitly asymmetric time interface.

    Pitch retains the cyclic tonic frame. Time instead combines a smooth
    monotonic bar coordinate, nested metrical phase, and directed pairwise
    distances. No source-song endpoint is accepted, so normalized progress
    cannot leak future termination.
    """

    def __init__(
        self,
        *,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
        frame_width: int = 16,
        motion_width: int = 68,
        realization_width: int = 40,
        ordered_progress_hidden_width: int = 240,
    ) -> None:
        super().__init__(
            use_realization=True,
            width=width,
            heads=heads,
            layers=layers,
            feedforward_width=feedforward_width,
            dropout=dropout,
            maximum_notes_per_group=maximum_notes_per_group,
            frame_width=frame_width,
            motion_width=motion_width,
            realization_width=realization_width,
        )

        # Replace the strong per-bar lookup with a smooth ordered coordinate.
        # The hidden width keeps the overall parameter budget close to B2
        # while every parameter participates in the time representation.
        del self.motion_absolute_bar
        self.motion_ordered_progress = nn.Sequential(
            nn.Linear(3, ordered_progress_hidden_width),
            nn.SiLU(),
            nn.Linear(ordered_progress_hidden_width, motion_width),
        )
        self.motion_bar_phase = nn.Linear(2, motion_width)

        # Directed coarse distances supplement, rather than replace, the
        # established exact/log onset-distance and within-bar relation paths.
        self.bar_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.beat_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        nn.init.zeros_(self.bar_distance_bias.weight)
        nn.init.zeros_(self.beat_distance_bias.weight)

    def musical_time_coordinates(
        self, melody: MelodyCausalBatch
    ) -> dict[str, torch.Tensor]:
        dtype = self.motion_bar_phase.weight.dtype
        bar = melody.context_absolute_bar.to(dtype)
        scale = float(max(M4E_MAX_ABSOLUTE_BARS - 1, 1))
        normalized = bar / scale
        ordered = torch.stack(
            (
                normalized,
                torch.log1p(bar) / math.log(float(M4E_MAX_ABSOLUTE_BARS)),
                torch.sqrt(normalized.clamp_min(0.0)),
            ),
            dim=-1,
        )
        within_bar = torch.remainder(
            melody.context_anchor_ticks, M4E_TICKS_PER_SEMANTIC_BAR
        )
        bar_phase = cyclic_phase(
            within_bar, M4E_TICKS_PER_SEMANTIC_BAR, dtype=dtype
        )
        beat_phase = cyclic_phase(
            melody.context_beat,
            M4E_BEATS_PER_SEMANTIC_BAR,
            dtype=dtype,
        )
        subdivision_phase = cyclic_phase(
            melody.context_fine_tick, M4E_TICKS_PER_BEAT, dtype=dtype
        )
        valid = melody.context_valid.unsqueeze(-1)
        return {
            "ordered_progress": ordered.masked_fill(~valid, 0.0),
            "bar_phase": bar_phase.masked_fill(~valid, 0.0),
            "beat_phase": beat_phase.masked_fill(~valid, 0.0),
            "subdivision_phase": subdivision_phase.masked_fill(~valid, 0.0),
        }

    def relative_motion_stream(
        self, batch: TonicCoordinateBatch
    ) -> torch.Tensor:
        melody = batch.melody
        dtype = self.motion_relative_phase.weight.dtype
        relative_pc = torch.remainder(
            melody.context_pitches - batch.tonic[:, None, None], 12
        )
        relative_phase = cyclic_phase(relative_pc, 12, dtype=dtype)
        gap_from_anchor = (
            melody.context_pitches - melody.context_pitches[:, :, :1]
        ).clamp(min=0, max=127)
        per_note = (
            self.motion_relative_phase(relative_phase)
            + self.motion_duration_bars(melody.context_duration_bars)
            + self.motion_duration_remainders(
                melody.context_duration_remainders
            )
            + self.motion_gap(gap_from_anchor)
        )
        note_mask = melody.context_note_mask.unsqueeze(-1).to(per_note.dtype)
        pooled = (per_note * note_mask).sum(dim=2) / note_mask.sum(
            dim=2
        ).clamp_min(1.0)
        pooled = pooled + self.motion_cardinality(melody.context_cardinality)

        rest = self.motion_rest_bars(
            melody.context_rest_bars
        ) + self.motion_rest_remainders(melody.context_rest_remainders)
        content = torch.zeros_like(pooled)
        content = torch.where(
            (melody.context_type == INPUT_NOTE_GROUP).unsqueeze(-1),
            pooled,
            content,
        )
        content = torch.where(
            (melody.context_type == INPUT_REST).unsqueeze(-1), rest, content
        )

        raw_interval = melody.local_pitch_relation - PITCH_DELTA_OFFSET
        pitched = melody.local_pitch_relation != PITCH_RELATION_NEUTRAL
        interval = torch.where(
            pitched, raw_interval, torch.zeros_like(raw_interval)
        ).to(dtype)
        interval_features = torch.stack(
            (
                interval / 127.0,
                interval.sign(),
                interval.abs() / 127.0,
                torch.cos(interval * (2.0 * math.pi / 12.0)),
                torch.sin(interval * (2.0 * math.pi / 12.0)),
            ),
            dim=-1,
        )
        interval_features = interval_features * pitched.unsqueeze(-1).to(dtype)

        time = self.musical_time_coordinates(melody)
        musical_time = self.project_musical_time(time)
        values = (
            self.motion_type(melody.context_type)
            + content
            + self.motion_local_kind(melody.local_kind)
            + self.motion_local_time_bars(melody.local_time_bars)
            + self.motion_local_time_remainders(
                melody.local_time_remainders
            )
            + self.motion_local_interval(interval_features)
            + musical_time
        )
        values = self.motion_projection(values)
        return values.masked_fill(~melody.context_valid.unsqueeze(-1), 0.0)

    def project_musical_time(
        self, time: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Project the frozen Step-1 time coordinates.

        This seam keeps historical behavior bit-identical while allowing a
        later carrier to replace absolute progress without copying the full
        pitch/time motion encoder.
        """

        meter = self.motion_meter_phase(
            torch.cat(
                (time["beat_phase"], time["subdivision_phase"]), dim=-1
            )
        )
        return (
            self.motion_ordered_progress(time["ordered_progress"])
            + self.motion_bar_phase(time["bar_phase"])
            + meter
        )

    def relative_attention_bias(
        self, batch: MelodyCausalBatch
    ) -> torch.Tensor:
        time_delta = (
            batch.context_anchor_ticks[:, :, None]
            - batch.context_anchor_ticks[:, None, :]
        )
        onset = self.time_bias(signed_log_bucket(time_delta))
        meter = self.meter_bias(
            torch.remainder(time_delta, M4E_TICKS_PER_SEMANTIC_BAR)
        )

        delta_bar = signed_unit_distance(
            time_delta, M4E_TICKS_PER_SEMANTIC_BAR
        )
        delta_beat = signed_unit_distance(time_delta, M4E_TICKS_PER_BEAT)
        bar = self.bar_distance_bias(
            signed_log_bucket(
                delta_bar,
                exact_magnitudes=8,
                maximum_distance=M4E_MAX_ABSOLUTE_BARS,
            )
        )
        beat = self.beat_distance_bias(
            signed_log_bucket(
                delta_beat,
                exact_magnitudes=16,
                maximum_distance=(
                    M4E_MAX_ABSOLUTE_BARS * M4E_BEATS_PER_SEMANTIC_BAR
                ),
            )
        )

        pitch_delta = (
            batch.context_anchor_pitch[:, :, None]
            - batch.context_anchor_pitch[:, None, :]
        ).clamp(min=-127, max=127)
        pitched = (
            (batch.context_type[:, :, None] == INPUT_NOTE_GROUP)
            & (batch.context_type[:, None, :] == INPUT_NOTE_GROUP)
        )
        pitch_index = torch.where(
            pitched,
            pitch_delta + PITCH_DELTA_OFFSET,
            torch.full_like(pitch_delta, PITCH_RELATION_NEUTRAL),
        )
        pitch = self.pitch_bias(pitch_index)
        return (onset + meter + bar + beat + pitch).permute(0, 3, 1, 2)


class Phase4BStep1MelodyModel(nn.Module):
    """Uniform baseline/candidate interface for the Step 1 direction probe."""

    def __init__(
        self,
        *,
        phase4b_arm: str,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
        frame_width: int = 16,
        motion_width: int = 68,
        realization_width: int = 40,
        ordered_progress_hidden_width: int = 240,
    ) -> None:
        super().__init__()
        if phase4b_arm not in PHASE4B_STEP1_ARMS:
            raise ValueError(f"unsupported Phase 4B Step 1 arm: {phase4b_arm}")
        self.phase4b_arm = phase4b_arm
        # Reused by the frozen Phase 4A.1 training/evaluation helpers.
        self.phase4a1_arm = phase4b_arm
        self.width = int(width)
        common = dict(
            width=width,
            heads=heads,
            layers=layers,
            feedforward_width=feedforward_width,
            dropout=dropout,
            maximum_notes_per_group=maximum_notes_per_group,
            frame_width=frame_width,
            motion_width=motion_width,
            realization_width=realization_width,
        )
        if phase4b_arm == PHASE4B_STEP1_BASELINE:
            self.model = TonalFrameFactorizedCore(
                use_realization=True, **common
            )
        else:
            self.model = MusicalTimeFactorizedCore(
                ordered_progress_hidden_width=ordered_progress_hidden_width,
                **common,
            )

    @property
    def decoder(self) -> FactorizedNextUnitDecoder:
        return self.model.decoder

    @property
    def uses_musical_time_geometry(self) -> bool:
        return self.phase4b_arm == PHASE4B_STEP1_MUSICAL_TIME

    def causal_hidden(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        return self.model.causal_hidden(batch)

    def losses(self, batch: TonicCoordinateBatch) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.causal_hidden(batch), batch.melody)

    def encode_streams(
        self, batch: TonicCoordinateBatch
    ) -> dict[str, torch.Tensor]:
        return self.model.encode_streams(batch)

    def state_taps(self, batch: TonicCoordinateBatch) -> dict[str, torch.Tensor]:
        return self.model.forward_with_taps(batch)

    def tonic_coordinates(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        return tonic_so2_coordinates(
            batch.tonic, dtype=next(self.parameters()).dtype
        )
