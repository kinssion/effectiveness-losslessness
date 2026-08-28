from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal

import numpy as np
import torch
from torch import nn


M4E_DISCRETE_PRIMITIVE_SCHEMA: Final = (
    "usmm.discrete_music_primitive.m4e.preflight.v1"
)
M4E_TICKS_PER_BEAT: Final = 96
M4E_BEATS_PER_SEMANTIC_BAR: Final = 4
M4E_TICKS_PER_SEMANTIC_BAR: Final = (
    M4E_TICKS_PER_BEAT * M4E_BEATS_PER_SEMANTIC_BAR
)
M4E_MAX_ABSOLUTE_BARS: Final = 256
M4E_MAX_CONTEXT_BARS: Final = 32
M4E_MAX_DURATION_BARS: Final = 256

PrimitiveArm = Literal[
    "A_RAW_EVENT_SEQUENCE",
    "B_FACTORIZED_PRIMITIVE",
    "C_FACTORIZED_PRIMITIVE_R1",
]


@dataclass(frozen=True, slots=True)
class CanonicalPrimitiveSong:
    song_id: str
    split: str
    bar_count: int
    onsets: np.ndarray
    pitches: np.ndarray
    durations: np.ndarray
    velocities: np.ndarray
    programs: np.ndarray
    track_entities: np.ndarray
    source_ids: np.ndarray
    source_relative_path: str
    source_sha256: str
    schema_version: str = M4E_DISCRETE_PRIMITIVE_SCHEMA

    @property
    def event_count(self) -> int:
        return int(self.onsets.shape[0])

    def assert_contract(self) -> None:
        if self.schema_version != M4E_DISCRETE_PRIMITIVE_SCHEMA:
            raise AssertionError("unsupported canonical primitive song schema")
        if not self.song_id or self.split not in {"train", "development"}:
            raise AssertionError("canonical primitive song identity is invalid")
        if not 0 < self.bar_count < M4E_MAX_ABSOLUTE_BARS:
            raise AssertionError("canonical primitive song bar support is invalid")
        # A full song can be longer than the model's context support. Validate
        # occurrence arrays directly while retaining the stricter window cap.
        arrays = (
            self.onsets,
            self.pitches,
            self.durations,
            self.velocities,
            self.programs,
            self.track_entities,
            self.source_ids,
        )
        if any(value.ndim != 1 for value in arrays) or len(
            {value.shape[0] for value in arrays}
        ) != 1:
            raise AssertionError("canonical primitive song arrays differ")
        if self.event_count:
            if int(self.onsets.min()) < 0 or int(self.onsets.max()) >= (
                self.bar_count * M4E_TICKS_PER_SEMANTIC_BAR
            ):
                raise AssertionError("canonical primitive song onset leaves support")
            if int(self.pitches.min()) < 0 or int(self.pitches.max()) > 127:
                raise AssertionError("canonical primitive song pitch leaves support")
            if bool(np.any(self.durations <= 0)):
                raise AssertionError("canonical primitive song duration must be positive")
            if int(self.durations.max()) >= (
                M4E_MAX_DURATION_BARS * M4E_TICKS_PER_SEMANTIC_BAR
            ):
                raise AssertionError("canonical primitive duration leaves support")
            if int(self.velocities.min()) < 1 or int(self.velocities.max()) > 127:
                raise AssertionError("canonical primitive velocity leaves MIDI support")
            if int(self.programs.min()) < 0 or int(self.programs.max()) > 127:
                raise AssertionError("canonical primitive program leaves MIDI support")
        if len(self.source_sha256) != 64:
            raise AssertionError("canonical primitive source hash is invalid")

    def window(self, start_bar: int, end_bar: int) -> "PrimitiveEventWindow":
        self.assert_contract()
        if not 0 <= start_bar < end_bar <= self.bar_count:
            raise ValueError("primitive song window leaves support")
        if end_bar - start_bar > M4E_MAX_CONTEXT_BARS:
            raise ValueError("primitive song window exceeds model context support")
        left = start_bar * M4E_TICKS_PER_SEMANTIC_BAR
        right = end_bar * M4E_TICKS_PER_SEMANTIC_BAR
        selected = (self.onsets >= left) & (self.onsets < right)
        result = PrimitiveEventWindow(
            onsets=(self.onsets[selected] - left).astype(np.int64, copy=True),
            pitches=self.pitches[selected].astype(np.int16, copy=True),
            durations=self.durations[selected].astype(np.int64, copy=True),
            velocities=self.velocities[selected].astype(np.uint8, copy=True),
            programs=self.programs[selected].astype(np.uint8, copy=True),
            track_entities=self.track_entities[selected].astype(np.int16, copy=True),
            source_ids=self.source_ids[selected].astype(np.int64, copy=True),
            absolute_start_bar=start_bar,
            context_bars=end_bar - start_bar,
        )
        result.assert_contract()
        return result

    def save_npz(self, path: object) -> None:
        from pathlib import Path

        target = Path(path)
        self.assert_contract()
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            schema_version=np.asarray(self.schema_version),
            song_id=np.asarray(self.song_id),
            split=np.asarray(self.split),
            bar_count=np.asarray(self.bar_count, dtype=np.int16),
            onsets=self.onsets,
            pitches=self.pitches,
            durations=self.durations,
            velocities=self.velocities,
            programs=self.programs,
            track_entities=self.track_entities,
            source_ids=self.source_ids,
            source_relative_path=np.asarray(self.source_relative_path),
            source_sha256=np.asarray(self.source_sha256),
        )

    @classmethod
    def load_npz(cls, path: object) -> "CanonicalPrimitiveSong":
        with np.load(path, allow_pickle=False) as payload:
            result = cls(
                song_id=str(payload["song_id"].item()),
                split=str(payload["split"].item()),
                bar_count=int(payload["bar_count"].item()),
                onsets=payload["onsets"].copy(),
                pitches=payload["pitches"].copy(),
                durations=payload["durations"].copy(),
                velocities=payload["velocities"].copy(),
                programs=payload["programs"].copy(),
                track_entities=payload["track_entities"].copy(),
                source_ids=payload["source_ids"].copy(),
                source_relative_path=str(payload["source_relative_path"].item()),
                source_sha256=str(payload["source_sha256"].item()),
                schema_version=str(payload["schema_version"].item()),
            )
        result.assert_contract()
        return result


@dataclass(frozen=True, slots=True)
class PrimitiveEventWindow:
    """Canonical score-event occurrences within one registered container.

    ``onsets`` are relative to ``absolute_start_bar`` and expressed in the
    fixed canonical score lattice. Rows are storage only: ``source_ids`` carry
    provenance and no model component may use their array positions.
    """

    onsets: np.ndarray
    pitches: np.ndarray
    durations: np.ndarray
    velocities: np.ndarray
    programs: np.ndarray
    track_entities: np.ndarray
    source_ids: np.ndarray
    absolute_start_bar: int
    context_bars: int
    schema_version: str = M4E_DISCRETE_PRIMITIVE_SCHEMA

    @property
    def event_count(self) -> int:
        return int(self.onsets.shape[0])

    @property
    def context_ticks(self) -> int:
        return int(self.context_bars * M4E_TICKS_PER_SEMANTIC_BAR)

    def assert_contract(self) -> None:
        if self.schema_version != M4E_DISCRETE_PRIMITIVE_SCHEMA:
            raise AssertionError("unsupported discrete primitive schema")
        arrays = (
            self.onsets,
            self.pitches,
            self.durations,
            self.velocities,
            self.programs,
            self.track_entities,
            self.source_ids,
        )
        if any(value.ndim != 1 for value in arrays):
            raise AssertionError("primitive event arrays must be one-dimensional")
        if len({value.shape[0] for value in arrays}) != 1:
            raise AssertionError("primitive event arrays differ in length")
        if not 0 <= self.absolute_start_bar < M4E_MAX_ABSOLUTE_BARS:
            raise AssertionError("absolute start bar leaves frozen support")
        if not 0 < self.context_bars <= M4E_MAX_CONTEXT_BARS:
            raise AssertionError("primitive context leaves frozen support")
        if self.event_count:
            if int(self.onsets.min()) < 0 or int(self.onsets.max()) >= self.context_ticks:
                raise AssertionError("primitive onset leaves context support")
            if int(self.pitches.min()) < 0 or int(self.pitches.max()) > 127:
                raise AssertionError("primitive pitch leaves MIDI support")
            if bool(np.any(self.durations <= 0)):
                raise AssertionError("primitive duration must be positive")
            if int(self.durations.max()) >= (
                M4E_MAX_DURATION_BARS * M4E_TICKS_PER_SEMANTIC_BAR
            ):
                raise AssertionError("primitive duration leaves frozen support")
            if int(self.velocities.min()) < 1 or int(self.velocities.max()) > 127:
                raise AssertionError("primitive velocity leaves MIDI support")
            if int(self.programs.min()) < 0 or int(self.programs.max()) > 127:
                raise AssertionError("primitive program leaves MIDI support")

    def permuted(self, order: np.ndarray) -> "PrimitiveEventWindow":
        self.assert_contract()
        if order.shape != (self.event_count,) or set(order.tolist()) != set(
            range(self.event_count)
        ):
            raise ValueError("primitive event permutation is invalid")
        return PrimitiveEventWindow(
            onsets=self.onsets[order].copy(),
            pitches=self.pitches[order].copy(),
            durations=self.durations[order].copy(),
            velocities=self.velocities[order].copy(),
            programs=self.programs[order].copy(),
            track_entities=self.track_entities[order].copy(),
            source_ids=self.source_ids[order].copy(),
            absolute_start_bar=self.absolute_start_bar,
            context_bars=self.context_bars,
        )

    def canonicalized(self) -> "PrimitiveEventWindow":
        self.assert_contract()
        if not self.event_count:
            return self
        order = np.lexsort(
            (
                self.source_ids,
                self.track_entities,
                self.durations,
                self.pitches,
                self.onsets,
            )
        )
        return self.permuted(order.astype(np.int64))

    def transposed(self, semitones: int) -> "PrimitiveEventWindow":
        pitches = self.pitches.astype(np.int16) + int(semitones)
        if pitches.size and (int(pitches.min()) < 0 or int(pitches.max()) > 127):
            raise ValueError("primitive transpose leaves MIDI support")
        return PrimitiveEventWindow(
            onsets=self.onsets.copy(),
            pitches=pitches.astype(np.int16),
            durations=self.durations.copy(),
            velocities=self.velocities.copy(),
            programs=self.programs.copy(),
            track_entities=self.track_entities.copy(),
            source_ids=self.source_ids.copy(),
            absolute_start_bar=self.absolute_start_bar,
            context_bars=self.context_bars,
        )

    def translated(self, bars: int) -> "PrimitiveEventWindow":
        start = self.absolute_start_bar + int(bars)
        if not 0 <= start < M4E_MAX_ABSOLUTE_BARS:
            raise ValueError("primitive time translation leaves absolute support")
        return PrimitiveEventWindow(
            onsets=self.onsets.copy(),
            pitches=self.pitches.copy(),
            durations=self.durations.copy(),
            velocities=self.velocities.copy(),
            programs=self.programs.copy(),
            track_entities=self.track_entities.copy(),
            source_ids=self.source_ids.copy(),
            absolute_start_bar=start,
            context_bars=self.context_bars,
        )


@dataclass(frozen=True, slots=True)
class DiscretePrimitiveBatch:
    absolute_pitch: torch.Tensor
    relative_pitch: torch.Tensor
    relative_bar: torch.Tensor
    beat_index: torch.Tensor
    fine_tick: torch.Tensor
    duration_bars: torch.Tensor
    duration_remainder: torch.Tensor
    relative_onset: torch.Tensor
    absolute_onset: torch.Tensor
    duration_ticks: torch.Tensor
    event_mask: torch.Tensor
    pitch_anchor: torch.Tensor
    absolute_start_bar: torch.Tensor

    def to(self, device: torch.device | str) -> "DiscretePrimitiveBatch":
        return DiscretePrimitiveBatch(
            absolute_pitch=self.absolute_pitch.to(device),
            relative_pitch=self.relative_pitch.to(device),
            relative_bar=self.relative_bar.to(device),
            beat_index=self.beat_index.to(device),
            fine_tick=self.fine_tick.to(device),
            duration_bars=self.duration_bars.to(device),
            duration_remainder=self.duration_remainder.to(device),
            relative_onset=self.relative_onset.to(device),
            absolute_onset=self.absolute_onset.to(device),
            duration_ticks=self.duration_ticks.to(device),
            event_mask=self.event_mask.to(device),
            pitch_anchor=self.pitch_anchor.to(device),
            absolute_start_bar=self.absolute_start_bar.to(device),
        )

    @property
    def shape(self) -> torch.Size:
        return self.absolute_pitch.shape

    def assert_contract(self) -> None:
        shape = self.absolute_pitch.shape
        if len(shape) != 2:
            raise AssertionError("primitive batch must be [batch,events]")
        for name in (
            "relative_pitch",
            "relative_bar",
            "beat_index",
            "fine_tick",
            "duration_bars",
            "duration_remainder",
            "relative_onset",
            "absolute_onset",
            "duration_ticks",
            "event_mask",
        ):
            if getattr(self, name).shape != shape:
                raise AssertionError(f"primitive batch field differs: {name}")
        if self.pitch_anchor.shape != shape[:1] or self.absolute_start_bar.shape != shape[:1]:
            raise AssertionError("primitive absolute context shape differs")
        if self.event_mask.dtype != torch.bool:
            raise AssertionError("primitive event mask must be bool")
        active = self.event_mask
        if bool(active.any()):
            if bool((self.absolute_pitch[active] < 0).any()) or bool(
                (self.absolute_pitch[active] > 127).any()
            ):
                raise AssertionError("primitive pitch id leaves support")
            if bool((self.fine_tick[active] < 0).any()) or bool(
                (self.fine_tick[active] >= M4E_TICKS_PER_BEAT).any()
            ):
                raise AssertionError("primitive fine tick leaves support")
            if bool((self.duration_ticks[active] <= 0).any()):
                raise AssertionError("primitive duration must be positive")


def collate_primitive_windows(
    windows: list[PrimitiveEventWindow],
    *,
    maximum_events: int | None = None,
) -> DiscretePrimitiveBatch:
    if not windows:
        raise ValueError("primitive collate requires at least one window")
    for window in windows:
        window.assert_contract()
    lengths = [
        min(window.event_count, maximum_events)
        if maximum_events is not None
        else window.event_count
        for window in windows
    ]
    maximum = max(max(lengths), 1)
    shape = (len(windows), maximum)
    arrays = {
        name: np.zeros(shape, dtype=np.int64)
        for name in (
            "absolute_pitch",
            "relative_pitch",
            "relative_bar",
            "beat_index",
            "fine_tick",
            "duration_bars",
            "duration_remainder",
            "relative_onset",
            "absolute_onset",
            "duration_ticks",
        )
    }
    mask = np.zeros(shape, dtype=bool)
    anchors = np.zeros(len(windows), dtype=np.int64)
    starts = np.zeros(len(windows), dtype=np.int64)
    for row, (window, length) in enumerate(zip(windows, lengths)):
        starts[row] = window.absolute_start_bar
        if not length:
            anchors[row] = 60
            continue
        # A cap is selected in musical-time order. Uncapped windows preserve
        # caller storage order so permutation-neutrality is genuinely tested.
        canonical = window.canonicalized() if length < window.event_count else window
        selected = slice(0, length)
        onset = canonical.onsets[selected].astype(np.int64)
        pitch = canonical.pitches[selected].astype(np.int64)
        duration = canonical.durations[selected].astype(np.int64)
        # Integer transposition must move the anchor by exactly the same
        # number of semitones. ``floor(mean)`` has that property; banker's
        # rounding does not at half-integer boundaries.
        anchor = int(np.floor(float(pitch.mean())))
        anchors[row] = anchor
        arrays["absolute_pitch"][row, :length] = pitch
        arrays["relative_pitch"][row, :length] = np.clip(pitch - anchor + 128, 0, 256)
        arrays["relative_bar"][row, :length] = np.minimum(
            onset // M4E_TICKS_PER_SEMANTIC_BAR, M4E_MAX_CONTEXT_BARS
        )
        local = onset % M4E_TICKS_PER_SEMANTIC_BAR
        arrays["beat_index"][row, :length] = local // M4E_TICKS_PER_BEAT
        arrays["fine_tick"][row, :length] = local % M4E_TICKS_PER_BEAT
        arrays["duration_bars"][row, :length] = duration // M4E_TICKS_PER_SEMANTIC_BAR
        arrays["duration_remainder"][row, :length] = (
            duration % M4E_TICKS_PER_SEMANTIC_BAR
        )
        arrays["relative_onset"][row, :length] = onset
        arrays["absolute_onset"][row, :length] = (
            window.absolute_start_bar * M4E_TICKS_PER_SEMANTIC_BAR + onset
        )
        arrays["duration_ticks"][row, :length] = duration
        mask[row, :length] = True
    result = DiscretePrimitiveBatch(
        **{name: torch.from_numpy(value) for name, value in arrays.items()},
        event_mask=torch.from_numpy(mask),
        pitch_anchor=torch.from_numpy(anchors),
        absolute_start_bar=torch.from_numpy(starts),
    )
    result.assert_contract()
    return result


_TIME_BUCKET_EDGES: Final = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    192,
    384,
    768,
    1536,
)
M4E_DELTA_TIME_CLASSES: Final = 1 + 2 * (len(_TIME_BUCKET_EDGES) + 1)
M4E_DELTA_PITCH_CLASSES: Final = 51
M4E_DURATION_RATIO_CLASSES: Final = 10
M4E_OVERLAP_CLASSES: Final = 8
M4E_METRICAL_CLASSES: Final = 8


@dataclass(frozen=True, slots=True)
class DiscreteRelationIds:
    delta_pitch: torch.Tensor
    delta_time: torch.Tensor
    same_onset: torch.Tensor
    duration_ratio: torch.Tensor
    overlap: torch.Tensor
    metrical: torch.Tensor
    pair_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DiscreteRelationIds":
        return DiscreteRelationIds(
            delta_pitch=self.delta_pitch.to(device),
            delta_time=self.delta_time.to(device),
            same_onset=self.same_onset.to(device),
            duration_ratio=self.duration_ratio.to(device),
            overlap=self.overlap.to(device),
            metrical=self.metrical.to(device),
            pair_mask=self.pair_mask.to(device),
        )


def _signed_time_bucket(delta: torch.Tensor) -> torch.Tensor:
    edges = torch.tensor(_TIME_BUCKET_EDGES, device=delta.device, dtype=delta.dtype)
    magnitude = torch.bucketize(delta.abs().contiguous(), edges)
    signed = 1 + 2 * magnitude + (delta > 0).to(torch.long)
    return torch.where(delta == 0, torch.zeros_like(signed), signed)


def discrete_relation_ids(batch: DiscretePrimitiveBatch) -> DiscreteRelationIds:
    batch.assert_contract()
    pitch_left = batch.absolute_pitch[:, :, None]
    pitch_right = batch.absolute_pitch[:, None, :]
    onset_left = batch.relative_onset[:, :, None]
    onset_right = batch.relative_onset[:, None, :]
    duration_left = batch.duration_ticks[:, :, None]
    duration_right = batch.duration_ticks[:, None, :]
    offset_left = onset_left + duration_left
    offset_right = onset_right + duration_right
    pair_mask = batch.event_mask[:, :, None] & batch.event_mask[:, None, :]

    delta_pitch_raw = pitch_right - pitch_left
    delta_pitch = torch.clamp(delta_pitch_raw, -24, 24) + 25
    delta_pitch = torch.where(
        delta_pitch_raw < -24,
        torch.zeros_like(delta_pitch),
        torch.where(delta_pitch_raw > 24, torch.full_like(delta_pitch, 50), delta_pitch),
    )
    delta_time = _signed_time_bucket(onset_right - onset_left)
    same_onset = (onset_left == onset_right).to(torch.long)

    ratio = torch.log2(
        duration_right.to(torch.float32).clamp_min(1.0)
        / duration_left.to(torch.float32).clamp_min(1.0)
    )
    ratio_edges = torch.tensor(
        (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0),
        device=ratio.device,
    )
    duration_ratio = torch.bucketize(ratio.contiguous(), ratio_edges)

    same_span = (onset_left == onset_right) & (offset_left == offset_right)
    left_contains = (onset_left <= onset_right) & (offset_left >= offset_right)
    right_contains = (onset_right <= onset_left) & (offset_right >= offset_left)
    partial_overlap = torch.maximum(onset_left, onset_right) < torch.minimum(
        offset_left, offset_right
    )
    overlap = torch.zeros_like(delta_time)
    overlap = torch.where(offset_left < onset_right, torch.zeros_like(overlap), overlap)
    overlap = torch.where(offset_left == onset_right, torch.ones_like(overlap), overlap)
    overlap = torch.where(partial_overlap, torch.full_like(overlap, 2), overlap)
    overlap = torch.where(left_contains, torch.full_like(overlap, 3), overlap)
    overlap = torch.where(right_contains, torch.full_like(overlap, 4), overlap)
    overlap = torch.where(same_span, torch.full_like(overlap, 5), overlap)
    overlap = torch.where(onset_left == offset_right, torch.full_like(overlap, 6), overlap)
    overlap = torch.where(onset_left > offset_right, torch.full_like(overlap, 7), overlap)

    fine_left = torch.remainder(onset_left, M4E_TICKS_PER_BEAT)
    fine_right = torch.remainder(onset_right, M4E_TICKS_PER_BEAT)
    beat_left = torch.div(onset_left, M4E_TICKS_PER_BEAT, rounding_mode="floor")
    beat_right = torch.div(onset_right, M4E_TICKS_PER_BEAT, rounding_mode="floor")
    metrical = (
        (fine_left == fine_right).to(torch.long)
        + 2
        * (
            torch.remainder(beat_left, M4E_BEATS_PER_SEMANTIC_BAR)
            == torch.remainder(beat_right, M4E_BEATS_PER_SEMANTIC_BAR)
        ).to(torch.long)
        + 4
        * (
            torch.remainder(onset_left, M4E_TICKS_PER_SEMANTIC_BAR)
            == torch.remainder(onset_right, M4E_TICKS_PER_SEMANTIC_BAR)
        ).to(torch.long)
    )

    def masked(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.long) * pair_mask.to(torch.long)

    return DiscreteRelationIds(
        delta_pitch=masked(delta_pitch),
        delta_time=masked(delta_time),
        same_onset=masked(same_onset),
        duration_ratio=masked(duration_ratio),
        overlap=masked(overlap),
        metrical=masked(metrical),
        pair_mask=pair_mask,
    )


class DiscreteRelationAttentionBias(nn.Module):
    def __init__(self, heads: int) -> None:
        super().__init__()
        self.delta_pitch = nn.Embedding(M4E_DELTA_PITCH_CLASSES, heads)
        self.delta_time = nn.Embedding(M4E_DELTA_TIME_CLASSES, heads)
        self.same_onset = nn.Embedding(2, heads)
        self.duration_ratio = nn.Embedding(M4E_DURATION_RATIO_CLASSES, heads)
        self.overlap = nn.Embedding(M4E_OVERLAP_CLASSES, heads)
        self.metrical = nn.Embedding(M4E_METRICAL_CLASSES, heads)

    def forward(self, relation: DiscreteRelationIds) -> torch.Tensor:
        value = (
            self.delta_pitch(relation.delta_pitch)
            + self.delta_time(relation.delta_time)
            + self.same_onset(relation.same_onset)
            + self.duration_ratio(relation.duration_ratio)
            + self.overlap(relation.overlap)
            + self.metrical(relation.metrical)
        )
        return value.permute(0, 3, 1, 2)


class _PrimitiveSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("primitive width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.relation_bias = DiscreteRelationAttentionBias(heads)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        relation: DiscreteRelationIds | None,
    ) -> torch.Tensor:
        batch, events, _ = values.shape
        qkv = self.qkv(values).view(batch, events, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_width)
        if relation is not None:
            scores = scores + self.relation_bias(relation)
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = torch.matmul(self.dropout(weights), value)
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch, events, self.width)
        return self.output(mixed) * mask[..., None].to(values.dtype)


class _PrimitiveEncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = _PrimitiveSelfAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        relation: DiscreteRelationIds | None,
    ) -> torch.Tensor:
        values = values + self.dropout(self.attention(self.norm1(values), mask, relation))
        values = values + self.dropout(self.ffn(self.norm2(values)))
        return values * mask[..., None].to(values.dtype)


@dataclass(frozen=True, slots=True)
class PrimitiveEncoderOutput:
    event_states: torch.Tensor
    relative_state: torch.Tensor
    absolute_context: torch.Tensor
    onset_group_states: torch.Tensor
    onset_group_onsets: torch.Tensor
    onset_group_mask: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight[..., None]).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)[
        ..., None
    ]


def _pool_onset_groups(
    values: torch.Tensor,
    onsets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[list[torch.Tensor]] = []
    onset_rows: list[list[torch.Tensor]] = []
    maximum = 1
    for batch_index in range(values.shape[0]):
        active_onsets = torch.unique(onsets[batch_index, mask[batch_index]], sorted=True)
        group_values: list[torch.Tensor] = []
        group_onsets: list[torch.Tensor] = []
        for onset in active_onsets:
            selected = mask[batch_index] & (onsets[batch_index] == onset)
            group_values.append(values[batch_index, selected].mean(dim=0))
            group_onsets.append(onset)
        rows.append(group_values)
        onset_rows.append(group_onsets)
        maximum = max(maximum, len(group_values))
    output = values.new_zeros(values.shape[0], maximum, values.shape[-1])
    output_onsets = onsets.new_zeros(values.shape[0], maximum)
    output_mask = torch.zeros(
        values.shape[0], maximum, dtype=torch.bool, device=values.device
    )
    for batch_index, (groups, group_onsets) in enumerate(zip(rows, onset_rows)):
        if groups:
            output[batch_index, : len(groups)] = torch.stack(groups)
            output_onsets[batch_index, : len(groups)] = torch.stack(group_onsets)
            output_mask[batch_index, : len(groups)] = True
    return output, output_onsets, output_mask


class DiscreteMusicPrimitiveEncoderM4E(nn.Module):
    """Small representation preflight encoder; not a Foundation Model."""

    def __init__(
        self,
        *,
        arm: PrimitiveArm,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if arm not in (
            "A_RAW_EVENT_SEQUENCE",
            "B_FACTORIZED_PRIMITIVE",
            "C_FACTORIZED_PRIMITIVE_R1",
        ):
            raise ValueError(f"unsupported primitive arm: {arm}")
        self.arm = arm
        self.width = width
        # Every arm owns the same parameter inventory. The arm changes only
        # the input path and whether the already-present relation table is read.
        self.raw_projection = nn.Sequential(nn.Linear(3, width), nn.LayerNorm(width))
        self.relative_pitch = nn.Embedding(257, width)
        self.relative_bar = nn.Embedding(M4E_MAX_CONTEXT_BARS + 1, width)
        self.beat = nn.Embedding(M4E_BEATS_PER_SEMANTIC_BAR, width)
        self.fine = nn.Embedding(M4E_TICKS_PER_BEAT, width)
        self.duration_bars = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.duration_remainder = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, width)
        self.factor_norm = nn.LayerNorm(width)
        self.pitch_anchor = nn.Embedding(128, width)
        self.absolute_bar = nn.Embedding(M4E_MAX_ABSOLUTE_BARS, width)
        self.absolute_projection = nn.Sequential(
            nn.Linear(width * 2, width), nn.LayerNorm(width)
        )
        self.blocks = nn.ModuleList(
            _PrimitiveEncoderBlock(width, heads, dropout) for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)

    def _content(self, batch: DiscretePrimitiveBatch) -> torch.Tensor:
        if self.arm == "A_RAW_EVENT_SEQUENCE":
            scale = float(M4E_MAX_ABSOLUTE_BARS * M4E_TICKS_PER_SEMANTIC_BAR)
            raw = torch.stack(
                (
                    batch.absolute_pitch.to(torch.float32) / 127.0,
                    batch.absolute_onset.to(torch.float32) / scale,
                    torch.log1p(batch.duration_ticks.to(torch.float32))
                    / math.log1p(M4E_MAX_DURATION_BARS * M4E_TICKS_PER_SEMANTIC_BAR),
                ),
                dim=-1,
            )
            return self.raw_projection(raw)
        factorized = (
            self.relative_pitch(batch.relative_pitch)
            + self.relative_bar(batch.relative_bar)
            + self.beat(batch.beat_index)
            + self.fine(batch.fine_tick)
            + self.duration_bars(batch.duration_bars)
            + self.duration_remainder(batch.duration_remainder)
        ) / math.sqrt(6.0)
        return self.factor_norm(factorized)

    def forward(self, batch: DiscretePrimitiveBatch) -> PrimitiveEncoderOutput:
        batch.assert_contract()
        values = self._content(batch) * batch.event_mask[..., None].to(torch.float32)
        relation = (
            discrete_relation_ids(batch)
            if self.arm == "C_FACTORIZED_PRIMITIVE_R1"
            else None
        )
        for block in self.blocks:
            values = block(values, batch.event_mask, relation)
        values = self.final_norm(values) * batch.event_mask[..., None].to(values.dtype)
        relative_state = _masked_mean(values, batch.event_mask)
        absolute_context = self.absolute_projection(
            torch.cat(
                (
                    self.pitch_anchor(batch.pitch_anchor.clamp(0, 127)),
                    self.absolute_bar(batch.absolute_start_bar.clamp(0, 255)),
                ),
                dim=-1,
            )
        )
        groups, group_onsets, group_mask = _pool_onset_groups(
            values, batch.relative_onset, batch.event_mask
        )
        return PrimitiveEncoderOutput(
            event_states=values,
            relative_state=relative_state,
            absolute_context=absolute_context,
            onset_group_states=groups,
            onset_group_onsets=group_onsets,
            onset_group_mask=group_mask,
        )


def primitive_representation_distance(
    left: PrimitiveEncoderOutput,
    right: PrimitiveEncoderOutput,
) -> float:
    return float(
        torch.linalg.vector_norm(left.relative_state - right.relative_state, dim=-1)
        .mean()
        .item()
    )


def primitive_absolute_context_distance(
    left: PrimitiveEncoderOutput,
    right: PrimitiveEncoderOutput,
) -> float:
    return float(
        torch.linalg.vector_norm(left.absolute_context - right.absolute_context, dim=-1)
        .mean()
        .item()
    )


def primitive_relation_consistency(
    left: DiscretePrimitiveBatch,
    right: DiscretePrimitiveBatch,
) -> float:
    a = discrete_relation_ids(left)
    b = discrete_relation_ids(right)
    fields = (
        "delta_pitch",
        "delta_time",
        "same_onset",
        "duration_ratio",
        "overlap",
        "metrical",
        "pair_mask",
    )
    scores = [
        float((getattr(a, name) == getattr(b, name)).to(torch.float32).mean().item())
        for name in fields
    ]
    return float(np.mean(scores))
