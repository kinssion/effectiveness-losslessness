from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Final, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .discrete_music_primitive import (
    M4E_MAX_ABSOLUTE_BARS,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .dual_coordinate_melody import (
    DualCoordinateMelodyModel,
    MelodyCausalBatch,
    signed_log_bucket,
)
from .sparse_melody_bpe import SparseMelodyStream


M4L_PHASE3_SCHEMA: Final = "usmm.m4l.harmonic_conditioning.phase3.v1"
M4L_PHASE3_ARMS: Final = (
    "C0_CAUSAL_MATCHED",
    "C1_CHORD_CROSS_ATTENTION",
)
CHORD_SLOTS_PER_BAR: Final = 2
CHORD_SLOT_TICKS: Final = M4E_TICKS_PER_SEMANTIC_BAR // CHORD_SLOTS_PER_BAR
NO_CHORD_ROOT: Final = 12
UNKNOWN_QUALITY: Final = "<UNK>"
NO_CHORD_QUALITY: Final = "N"

_PITCH_CLASS: Final = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def semantic_downbeats(path: Path) -> np.ndarray:
    """Read the exact POP909 downbeat map used by the frozen M4E carrier."""

    downbeats = [
        float(fields[0])
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(fields := line.split()) >= 3 and float(fields[2]) >= 0.5
    ]
    if len(downbeats) < 2:
        raise ValueError(f"insufficient POP909 downbeats: {path}")
    intervals = [
        right - left
        for left, right in zip(downbeats, downbeats[1:])
        if right > left
    ]
    if not intervals:
        raise ValueError(f"non-increasing POP909 downbeats: {path}")
    result = np.asarray(
        [*downbeats, downbeats[-1] + statistics.median(intervals)],
        dtype=np.float64,
    )
    if bool(np.any(result[1:] <= result[:-1])):
        raise ValueError(f"semantic downbeat map is not monotonic: {path}")
    return result


@dataclass(frozen=True, slots=True)
class ChordInterval:
    start_seconds: float
    end_seconds: float
    root_class: int
    quality: str

    def __post_init__(self) -> None:
        if not self.end_seconds > self.start_seconds:
            raise ValueError("chord interval must have positive duration")
        if not 0 <= self.root_class <= NO_CHORD_ROOT:
            raise ValueError("chord root leaves frozen support")


@dataclass(frozen=True, slots=True)
class CoarseChordSong:
    song_id: str
    root_classes: tuple[int, ...]
    quality_ids: tuple[int, ...]
    anchor_ticks: tuple[int, ...]
    absolute_bars: tuple[int, ...]
    half_bar_phases: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.root_classes),
            len(self.quality_ids),
            len(self.anchor_ticks),
            len(self.absolute_bars),
            len(self.half_bar_phases),
        }
        if len(lengths) != 1 or not self.root_classes:
            raise ValueError("coarse chord fields are not aligned")
        if any(
            right <= left
            for left, right in zip(self.anchor_ticks, self.anchor_ticks[1:])
        ):
            raise ValueError("coarse chord anchors must be strictly ordered")


@dataclass(slots=True)
class ChordConditionBatch:
    root_classes: torch.Tensor
    quality_ids: torch.Tensor
    anchor_ticks: torch.Tensor
    absolute_bars: torch.Tensor
    half_bar_phases: torch.Tensor
    valid: torch.Tensor

    def to(self, device: torch.device | str) -> "ChordConditionBatch":
        return ChordConditionBatch(
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "root_classes",
                    "quality_ids",
                    "anchor_ticks",
                    "absolute_bars",
                    "half_bar_phases",
                    "valid",
                )
            }
        )

    def index_select(self, rows: torch.Tensor) -> "ChordConditionBatch":
        rows = rows.to(device=self.root_classes.device, dtype=torch.long)
        return ChordConditionBatch(
            **{
                name: getattr(self, name).index_select(0, rows)
                for name in (
                    "root_classes",
                    "quality_ids",
                    "anchor_ticks",
                    "absolute_bars",
                    "half_bar_phases",
                    "valid",
                )
            }
        )


def parse_chord_label(label: str) -> tuple[int, str]:
    label = str(label).strip()
    if label == NO_CHORD_QUALITY:
        return NO_CHORD_ROOT, NO_CHORD_QUALITY
    if ":" not in label:
        raise ValueError(f"unsupported POP909 chord label: {label}")
    root, quality = label.split(":", 1)
    quality = quality.split("/", 1)[0]
    if root not in _PITCH_CLASS or not quality:
        raise ValueError(f"unsupported POP909 chord label: {label}")
    return _PITCH_CLASS[root], quality


def read_chord_intervals(path: Path) -> tuple[ChordInterval, ...]:
    intervals: list[ChordInterval] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        root, quality = parse_chord_label(fields[2])
        intervals.append(
            ChordInterval(
                start_seconds=float(fields[0]),
                end_seconds=float(fields[1]),
                root_class=root,
                quality=quality,
            )
        )
    if not intervals:
        raise ValueError(f"no chord intervals found: {path}")
    return tuple(intervals)


def fit_quality_vocabulary(
    chord_paths: Sequence[Path],
) -> tuple[str, ...]:
    qualities = {
        interval.quality
        for path in chord_paths
        for interval in read_chord_intervals(path)
    }
    qualities.discard(UNKNOWN_QUALITY)
    return (UNKNOWN_QUALITY, *sorted(qualities))


def build_coarse_chord_song(
    *,
    song_id: str,
    boundaries: np.ndarray,
    intervals: Sequence[ChordInterval],
    quality_to_id: Mapping[str, int],
) -> CoarseChordSong:
    bars = int(boundaries.shape[0] - 1)
    if bars <= 0 or bars > M4E_MAX_ABSOLUTE_BARS:
        raise ValueError("coarse chord song leaves bar support")
    roots: list[int] = []
    qualities: list[int] = []
    anchors: list[int] = []
    absolute_bars: list[int] = []
    phases: list[int] = []
    interval_index = 0
    ordered = tuple(intervals)
    for bar in range(bars):
        left = float(boundaries[bar])
        right = float(boundaries[bar + 1])
        for phase in range(CHORD_SLOTS_PER_BAR):
            midpoint = left + (phase + 0.5) / CHORD_SLOTS_PER_BAR * (right - left)
            while (
                interval_index + 1 < len(ordered)
                and ordered[interval_index].end_seconds <= midpoint
            ):
                interval_index += 1
            current = ordered[interval_index]
            if current.start_seconds <= midpoint < current.end_seconds:
                root = current.root_class
                quality = current.quality
            else:
                root = NO_CHORD_ROOT
                quality = NO_CHORD_QUALITY
            roots.append(root)
            qualities.append(int(quality_to_id.get(quality, 0)))
            anchors.append(bar * M4E_TICKS_PER_SEMANTIC_BAR + phase * CHORD_SLOT_TICKS)
            absolute_bars.append(bar)
            phases.append(phase)
    return CoarseChordSong(
        song_id=str(song_id),
        root_classes=tuple(roots),
        quality_ids=tuple(qualities),
        anchor_ticks=tuple(anchors),
        absolute_bars=tuple(absolute_bars),
        half_bar_phases=tuple(phases),
    )


def load_coarse_chord_song(
    *, song_id: str, song_directory: Path, quality_to_id: Mapping[str, int]
) -> CoarseChordSong:
    return build_coarse_chord_song(
        song_id=song_id,
        boundaries=semantic_downbeats(song_directory / "beat_midi.txt"),
        intervals=read_chord_intervals(song_directory / "chord_midi.txt"),
        quality_to_id=quality_to_id,
    )


def tensorize_chord_conditions(
    streams: Sequence[SparseMelodyStream],
    examples: Sequence[tuple[int, int]],
    songs: Mapping[str, CoarseChordSong],
    *,
    device: torch.device | str = "cpu",
) -> ChordConditionBatch:
    if not examples:
        raise ValueError("cannot tensorize zero chord-condition examples")
    selected = [songs[streams[stream_index].song_id] for stream_index, _ in examples]
    maximum = max(len(song.root_classes) for song in selected)
    shape = (len(selected), maximum)
    root = torch.full(shape, NO_CHORD_ROOT, dtype=torch.long)
    quality = torch.zeros(shape, dtype=torch.long)
    anchor = torch.zeros(shape, dtype=torch.long)
    bar = torch.zeros(shape, dtype=torch.long)
    phase = torch.zeros(shape, dtype=torch.long)
    valid = torch.zeros(shape, dtype=torch.bool)
    for row, song in enumerate(selected):
        length = len(song.root_classes)
        root[row, :length] = torch.tensor(song.root_classes, dtype=torch.long)
        quality[row, :length] = torch.tensor(song.quality_ids, dtype=torch.long)
        anchor[row, :length] = torch.tensor(song.anchor_ticks, dtype=torch.long)
        bar[row, :length] = torch.tensor(song.absolute_bars, dtype=torch.long)
        phase[row, :length] = torch.tensor(song.half_bar_phases, dtype=torch.long)
        valid[row, :length] = True
    return ChordConditionBatch(
        root_classes=root,
        quality_ids=quality,
        anchor_ticks=anchor,
        absolute_bars=bar,
        half_bar_phases=phase,
        valid=valid,
    ).to(device)


class LightweightChordCrossAttention(nn.Module):
    def __init__(self, *, width: int, condition_width: int) -> None:
        super().__init__()
        self.condition_width = int(condition_width)
        self.query_normalization = nn.LayerNorm(width)
        self.query = nn.Linear(width, condition_width)
        self.key = nn.Linear(condition_width, condition_width)
        self.value = nn.Linear(condition_width, condition_width)
        self.output = nn.Linear(condition_width, width)
        self.relative_time_bias = nn.Embedding(65, 1)
        nn.init.zeros_(self.relative_time_bias.weight)

    def residual(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        *,
        query_anchor_ticks: torch.Tensor,
        condition_anchor_ticks: torch.Tensor,
        query_valid: torch.Tensor,
        condition_valid: torch.Tensor,
        alignment_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query(self.query_normalization(values))
        key = self.key(condition)
        projected_value = self.value(condition)
        scores = torch.einsum("bld,bcd->blc", query, key)
        scores = scores / float(self.condition_width) ** 0.5
        delta = query_anchor_ticks[:, :, None] - condition_anchor_ticks[:, None, :]
        scores = scores + self.relative_time_bias(signed_log_bucket(delta)).squeeze(-1)
        allowed = condition_valid[:, None, :].expand_as(scores)
        if alignment_valid is not None:
            if alignment_valid.shape != scores.shape:
                raise ValueError("chord alignment mask shape differs")
            allowed = allowed & alignment_valid
        if bool((query_valid & ~allowed.any(dim=-1)).any()):
            raise ValueError("valid chord query has no aligned condition token")
        scores = scores.masked_fill(~allowed, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("blc,bcd->bld", attention, projected_value)
        return self.output(attended).masked_fill(~query_valid.unsqueeze(-1), 0.0)

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        *,
        query_anchor_ticks: torch.Tensor,
        condition_anchor_ticks: torch.Tensor,
        query_valid: torch.Tensor,
        condition_valid: torch.Tensor,
        alignment_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.residual(
            values,
            condition,
            query_anchor_ticks=query_anchor_ticks,
            condition_anchor_ticks=condition_anchor_ticks,
            query_valid=query_valid,
            condition_valid=condition_valid,
            alignment_valid=alignment_valid,
        )
        return values + residual


class HarmonicConditionedMelodyModel(DualCoordinateMelodyModel):
    """Full-A1 melody model with a matched optional coarse-chord path."""

    def __init__(
        self,
        *,
        phase3_arm: str,
        chord_quality_classes: int,
        condition_width: int = 16,
        **kwargs: object,
    ) -> None:
        if phase3_arm not in M4L_PHASE3_ARMS:
            raise ValueError(f"unsupported M4L Phase 3 arm: {phase3_arm}")
        super().__init__(arm="A1_DUAL_COORDINATE", **kwargs)
        self.phase3_arm = phase3_arm
        self.use_chord_condition = phase3_arm == "C1_CHORD_CROSS_ATTENTION"
        self.chord_root = nn.Embedding(13, condition_width)
        self.chord_quality = nn.Embedding(chord_quality_classes, condition_width)
        self.chord_bar = nn.Embedding(M4E_MAX_ABSOLUTE_BARS, condition_width)
        self.chord_half_bar = nn.Embedding(CHORD_SLOTS_PER_BAR, condition_width)
        self.chord_normalization = nn.LayerNorm(condition_width)
        self.chord_cross_attention = LightweightChordCrossAttention(
            width=self.width, condition_width=condition_width
        )

    def encode_chord_condition(self, chord: ChordConditionBatch) -> torch.Tensor:
        values = (
            self.chord_root(chord.root_classes)
            + self.chord_quality(chord.quality_ids)
            + self.chord_bar(chord.absolute_bars)
            + self.chord_half_bar(chord.half_bar_phases)
        )
        values = self.chord_normalization(values)
        return values.masked_fill(~chord.valid.unsqueeze(-1), 0.0)

    def causal_hidden_with_chords(
        self, batch: MelodyCausalBatch, chord: ChordConditionBatch
    ) -> torch.Tensor:
        if not self.use_chord_condition:
            return super().causal_hidden(batch)
        values = self.encode_inputs(batch)
        self_bias = self.relative_attention_bias(batch)
        chord_values = self.encode_chord_condition(chord)
        for block in self.blocks:
            values = block(
                values, valid=batch.context_valid, relative_bias=self_bias
            )
            values = self.chord_cross_attention(
                values,
                chord_values,
                query_anchor_ticks=batch.context_anchor_ticks,
                condition_anchor_ticks=chord.anchor_ticks,
                query_valid=batch.context_valid,
                condition_valid=chord.valid,
            )
        values = self.output_normalization(values)
        rows = torch.arange(batch.batch_size, device=values.device)
        return values[rows, batch.context_lengths - 1]

    def losses_with_chords(
        self, batch: MelodyCausalBatch, chord: ChordConditionBatch
    ) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.causal_hidden_with_chords(batch, chord), batch)
