from __future__ import annotations

import math
from typing import Final, Mapping

import torch
from torch import nn

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_ABSOLUTE_BARS,
    M4E_MAX_DURATION_BARS,
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
    DualCoordinateMelodyModel,
    FactorizedNextUnitDecoder,
    MelodyCausalBatch,
    RelativeCausalBlock,
    signed_log_bucket,
)
from .harmonic_conditioning_melody import ChordConditionBatch
from .harmonic_pitch_energy import CandidateWiseHarmonicEnergy
from .tonic_coordinate_melody import TonicCoordinateBatch


M4L_PHASE4A1_SCHEMA: Final = (
    "usmm.m4l.tonal_frame_realization_factorization.phase4a1.v1"
)
PHASE4A1_B0: Final = "B0_CURRENT_FULL_A1"
PHASE4A1_B1: Final = "B1_TONAL_FRAME_RELATIVE_DYNAMICS"
PHASE4A1_B2: Final = "B2_FULL_FACTORIZED"
PHASE4A1_ARMS: Final = (PHASE4A1_B0, PHASE4A1_B1, PHASE4A1_B2)


def tonic_so2_coordinates(
    tonic: torch.Tensor, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return the fixed two-dimensional real representation of Z12."""

    angle = tonic.to(dtype) * (2.0 * math.pi / 12.0)
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def rotate_so2(values: torch.Tensor, semitones: int) -> torch.Tensor:
    """Apply the known pitch-class rotation to row-vector SO(2) coordinates."""

    angle = float(semitones) * (2.0 * math.pi / 12.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = values.new_tensor(((cosine, sine), (-sine, cosine)))
    return values @ rotation


def _phase(values: torch.Tensor, period: int, dtype: torch.dtype) -> torch.Tensor:
    angle = values.to(dtype) * (2.0 * math.pi / float(period))
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


class TonalFrameFactorizedCore(nn.Module):
    """Three observable coordinate streams followed by one causal backbone.

    The streams intentionally answer different questions:

    * frame: the song-level cyclic tonic coordinate;
    * motion: pitch/time/duration relations expressed in that frame;
    * realization: the absolute physical pitch and causal tessitura.

    B1 and B2 share this exact topology.  B1 replaces the realization input
    with zero, rather than changing the model or the decoder.
    """

    def __init__(
        self,
        *,
        use_realization: bool,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
        frame_width: int = 16,
        motion_width: int = 68,
        realization_width: int = 40,
    ) -> None:
        super().__init__()
        self.use_realization = bool(use_realization)
        self.width = int(width)
        self.heads = int(heads)
        self.maximum_notes_per_group = int(maximum_notes_per_group)
        self.frame_width = int(frame_width)
        self.motion_width = int(motion_width)
        self.realization_width = int(realization_width)
        self.structural_width = self.frame_width + self.motion_width

        # Tonal frame.  The source is fixed SO(2), never a 12-way lookup.
        self.frame_encoder = nn.Sequential(
            nn.Linear(2, frame_width),
            nn.GELU(),
            nn.LayerNorm(frame_width),
        )

        # Relative dynamics.  Pitch class is a phase inside the supplied
        # frame; interval direction and magnitude are continuous coordinates.
        self.motion_type = nn.Embedding(4, motion_width)
        self.motion_relative_phase = nn.Linear(2, motion_width)
        self.motion_duration_bars = nn.Embedding(
            M4E_MAX_DURATION_BARS, motion_width
        )
        self.motion_duration_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, motion_width
        )
        self.motion_gap = nn.Embedding(128, motion_width)
        self.motion_cardinality = nn.Embedding(
            maximum_notes_per_group + 1, motion_width
        )
        self.motion_rest_bars = nn.Embedding(
            M4E_MAX_DURATION_BARS, motion_width
        )
        self.motion_rest_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, motion_width
        )
        self.motion_local_kind = nn.Embedding(6, motion_width)
        self.motion_local_time_bars = nn.Embedding(
            M4E_MAX_DURATION_BARS, motion_width
        )
        self.motion_local_time_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, motion_width
        )
        self.motion_local_interval = nn.Linear(5, motion_width)
        self.motion_meter_phase = nn.Linear(4, motion_width)
        self.motion_absolute_bar = nn.Embedding(
            M4E_MAX_ABSOLUTE_BARS, motion_width
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(motion_width),
            nn.Linear(motion_width, motion_width),
            nn.GELU(),
            nn.LayerNorm(motion_width),
        )

        # Physical realization.  Prefix range is computed causally below.
        self.realization_pitch = nn.Embedding(128, realization_width)
        self.realization_octave = nn.Embedding(11, realization_width)
        self.realization_range = nn.Linear(5, realization_width)
        self.realization_projection = nn.Sequential(
            nn.LayerNorm(realization_width),
            nn.Linear(realization_width, realization_width),
            nn.GELU(),
            nn.LayerNorm(realization_width),
        )

        fused_width = frame_width + motion_width + realization_width
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_width),
            nn.Linear(fused_width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )

        # The established Full-A1 pairwise time/pitch/meter biases remain.
        self.time_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.pitch_bias = nn.Embedding(PITCH_RELATION_CLASSES, heads)
        self.meter_bias = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, heads)
        for embedding in (self.time_bias, self.pitch_bias, self.meter_bias):
            nn.init.zeros_(embedding.weight)

        self.blocks = nn.ModuleList(
            RelativeCausalBlock(
                width=width,
                heads=heads,
                feedforward_width=feedforward_width,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.output_normalization = nn.LayerNorm(width)
        self.decoder = FactorizedNextUnitDecoder(
            width=width,
            maximum_notes_per_group=maximum_notes_per_group,
        )

    def tonal_frame_stream(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        source = tonic_so2_coordinates(
            batch.tonic, dtype=self.frame_encoder[0].weight.dtype
        )
        encoded = self.frame_encoder(source)[:, None, :]
        encoded = encoded.expand(-1, batch.melody.context_length, -1)
        return encoded.masked_fill(
            ~batch.melody.context_valid.unsqueeze(-1), 0.0
        )

    def relative_motion_stream(
        self, batch: TonicCoordinateBatch
    ) -> torch.Tensor:
        melody = batch.melody
        dtype = self.motion_relative_phase.weight.dtype
        relative_pc = torch.remainder(
            melody.context_pitches - batch.tonic[:, None, None], 12
        )
        relative_phase = _phase(relative_pc, 12, dtype)
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

        beat_phase = _phase(
            melody.context_beat, M4E_BEATS_PER_SEMANTIC_BAR, dtype
        )
        fine_phase = _phase(melody.context_fine_tick, M4E_TICKS_PER_BEAT, dtype)
        meter = self.motion_meter_phase(
            torch.cat((beat_phase, fine_phase), dim=-1)
        )
        values = (
            self.motion_type(melody.context_type)
            + content
            + self.motion_local_kind(melody.local_kind)
            + self.motion_local_time_bars(melody.local_time_bars)
            + self.motion_local_time_remainders(
                melody.local_time_remainders
            )
            + self.motion_local_interval(interval_features)
            + meter
            + self.motion_absolute_bar(melody.context_absolute_bar)
        )
        values = self.motion_projection(values)
        return values.masked_fill(~melody.context_valid.unsqueeze(-1), 0.0)

    @staticmethod
    def causal_tessitura_features(
        melody: MelodyCausalBatch, *, dtype: torch.dtype
    ) -> torch.Tensor:
        note_present = melody.context_note_mask.any(dim=2)
        step_min = melody.context_pitches.masked_fill(
            ~melody.context_note_mask, 127
        ).amin(dim=2)
        step_max = melody.context_pitches.masked_fill(
            ~melody.context_note_mask, 0
        ).amax(dim=2)
        prefix_min = torch.cummin(step_min, dim=1).values
        prefix_max = torch.cummax(step_max, dim=1).values
        seen = note_present.to(torch.long).cumsum(dim=1) > 0
        prefix_min = torch.where(seen, prefix_min, torch.zeros_like(prefix_min))
        prefix_max = torch.where(seen, prefix_max, torch.zeros_like(prefix_max))
        minimum = prefix_min.to(dtype) / 127.0
        maximum = prefix_max.to(dtype) / 127.0
        span = (prefix_max - prefix_min).to(dtype) / 127.0
        center = (prefix_max + prefix_min).to(dtype) / (2.0 * 127.0)
        return torch.stack(
            (minimum, maximum, span, center, seen.to(dtype)), dim=-1
        )

    def realization_stream(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        melody = batch.melody
        note_values = self.realization_pitch(
            melody.context_pitches
        ) + self.realization_octave(
            torch.div(melody.context_pitches, 12, rounding_mode="floor").clamp(
                min=0, max=10
            )
        )
        note_mask = melody.context_note_mask.unsqueeze(-1).to(note_values.dtype)
        pooled = (note_values * note_mask).sum(dim=2) / note_mask.sum(
            dim=2
        ).clamp_min(1.0)
        pooled = pooled.masked_fill(
            ~(melody.context_type == INPUT_NOTE_GROUP).unsqueeze(-1), 0.0
        )
        range_features = self.causal_tessitura_features(
            melody, dtype=note_values.dtype
        )
        values = self.realization_projection(
            pooled + self.realization_range(range_features)
        )
        values = values.masked_fill(~melody.context_valid.unsqueeze(-1), 0.0)
        if not self.use_realization:
            values = torch.zeros_like(values)
        return values

    def encode_streams(
        self, batch: TonicCoordinateBatch
    ) -> dict[str, torch.Tensor]:
        return {
            "frame": self.tonal_frame_stream(batch),
            "motion": self.relative_motion_stream(batch),
            "realization": self.realization_stream(batch),
        }

    def relative_attention_bias(
        self, batch: MelodyCausalBatch
    ) -> torch.Tensor:
        time_delta = (
            batch.context_anchor_ticks[:, :, None]
            - batch.context_anchor_ticks[:, None, :]
        )
        time = self.time_bias(signed_log_bucket(time_delta))
        meter = self.meter_bias(
            torch.remainder(time_delta, M4E_TICKS_PER_SEMANTIC_BAR)
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
        return (time + meter + self.pitch_bias(pitch_index)).permute(0, 3, 1, 2)

    def forward_with_taps(
        self, batch: TonicCoordinateBatch
    ) -> dict[str, torch.Tensor]:
        streams = self.encode_streams(batch)
        structural = torch.cat((streams["frame"], streams["motion"]), dim=-1)
        values = self.fusion(
            torch.cat((structural, streams["realization"]), dim=-1)
        )
        bias = self.relative_attention_bias(batch.melody)
        for block in self.blocks:
            values = block(
                values,
                valid=batch.melody.context_valid,
                relative_bias=bias,
            )
        values = self.output_normalization(values)
        rows = torch.arange(batch.batch_size, device=values.device)
        columns = batch.melody.context_lengths - 1
        return {
            "frame": streams["frame"][rows, columns],
            "motion": streams["motion"][rows, columns],
            "structure": structural[rows, columns],
            "realization": streams["realization"][rows, columns],
            "final": values[rows, columns],
        }

    def causal_hidden(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        return self.forward_with_taps(batch)["final"]

    def losses(self, batch: TonicCoordinateBatch) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.causal_hidden(batch), batch.melody)


class Phase4A1MelodyModel(nn.Module):
    """Uniform B0/B1/B2 interface without altering the established B0."""

    def __init__(
        self,
        *,
        phase4a1_arm: str,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
        frame_width: int = 16,
        motion_width: int = 68,
        realization_width: int = 40,
    ) -> None:
        super().__init__()
        if phase4a1_arm not in PHASE4A1_ARMS:
            raise ValueError(f"unsupported Phase 4A.1 arm: {phase4a1_arm}")
        self.phase4a1_arm = phase4a1_arm
        self.width = int(width)
        if phase4a1_arm == PHASE4A1_B0:
            self.model = DualCoordinateMelodyModel(
                arm="A1_DUAL_COORDINATE",
                width=width,
                heads=heads,
                layers=layers,
                feedforward_width=feedforward_width,
                dropout=dropout,
                maximum_notes_per_group=maximum_notes_per_group,
            )
        else:
            self.model = TonalFrameFactorizedCore(
                use_realization=phase4a1_arm == PHASE4A1_B2,
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

    @property
    def decoder(self) -> FactorizedNextUnitDecoder:
        return self.model.decoder

    @property
    def is_factorized(self) -> bool:
        return self.phase4a1_arm != PHASE4A1_B0

    def causal_hidden(self, batch: TonicCoordinateBatch) -> torch.Tensor:
        if self.is_factorized:
            return self.model.causal_hidden(batch)
        return self.model.causal_hidden(batch.melody)

    def state_taps(self, batch: TonicCoordinateBatch) -> dict[str, torch.Tensor]:
        if self.is_factorized:
            return self.model.forward_with_taps(batch)
        final = self.model.causal_hidden(batch.melody)
        return {"final": final, "structure": final}

    def encode_streams(
        self, batch: TonicCoordinateBatch
    ) -> dict[str, torch.Tensor]:
        if not self.is_factorized:
            raise ValueError("B0 has no separately observable factor streams")
        return self.model.encode_streams(batch)

    def losses(self, batch: TonicCoordinateBatch) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.causal_hidden(batch), batch.melody)


class FrozenPhase4A1HarmonicEnergyModel(nn.Module):
    """Frozen representation plus the established pitch-only energy adapter."""

    def __init__(
        self,
        *,
        base: Phase4A1MelodyModel,
        chord_quality_classes: int,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.energy = CandidateWiseHarmonicEnergy(
            width=self.base.width,
            chord_quality_classes=chord_quality_classes,
        )

    def adapter_parameters(self):
        return self.energy.parameters()

    def baseline_losses(
        self, batch: TonicCoordinateBatch
    ) -> Mapping[str, torch.Tensor]:
        hidden = self.base.causal_hidden(batch)
        return self.base.decoder.losses(hidden, batch.melody)

    def losses_with_chords(
        self,
        batch: TonicCoordinateBatch,
        chord: ChordConditionBatch,
    ) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
        hidden = self.base.causal_hidden(batch)
        bias, receipt = self.energy(hidden, batch.melody, chord)
        losses = self.base.decoder.losses(
            hidden, batch.melody, anchor_logit_bias=bias
        )
        return losses, receipt
