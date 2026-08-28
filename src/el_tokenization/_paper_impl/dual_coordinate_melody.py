from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_ABSOLUTE_BARS,
    M4E_MAX_DURATION_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .sparse_melody_bpe import (
    NOTE_GROUP,
    REST,
    MelodyAtom,
    SparseMelodyStream,
    SparseNote,
)


M4L_SCHEMA: Final = "usmm.m4l.dual_coordinate_melody.v1"
M4L_ARMS: Final = (
    "A0_ABSOLUTE_ONLY",
    "A_TIME_ONLY",
    "A1_DUAL_COORDINATE",
)

INPUT_PAD: Final = 0
INPUT_BOS: Final = 1
INPUT_NOTE_GROUP: Final = 2
INPUT_REST: Final = 3

TARGET_NOTE_GROUP: Final = 0
TARGET_REST: Final = 1
TARGET_EOS: Final = 2

TIME_NONE: Final = 0
TIME_ABSOLUTE_START: Final = 1
TIME_ONSET_DELTA: Final = 2
TIME_REST_DURATION: Final = 3

ANCHOR_NONE: Final = 0
ANCHOR_ABSOLUTE: Final = 1
ANCHOR_DELTA: Final = 2

LOCAL_NONE: Final = 0
LOCAL_START: Final = 1
LOCAL_NOTE_NOTE: Final = 2
LOCAL_NOTE_REST: Final = 3
LOCAL_REST_NOTE: Final = 4
LOCAL_REST_REST: Final = 5

PITCH_DELTA_OFFSET: Final = 127
PITCH_RELATION_NEUTRAL: Final = 255
PITCH_RELATION_CLASSES: Final = 256
TIME_RELATION_BUCKETS: Final = 65


def split_ticks(value: int) -> tuple[int, int]:
    value = int(value)
    if value < 0:
        raise ValueError("tick value must be non-negative")
    bars, remainder = divmod(value, M4E_TICKS_PER_SEMANTIC_BAR)
    if bars >= M4E_MAX_DURATION_BARS:
        raise ValueError("tick value leaves frozen hierarchical support")
    return bars, remainder


def join_ticks(bars: int, remainder: int) -> int:
    bars = int(bars)
    remainder = int(remainder)
    if not 0 <= bars < M4E_MAX_DURATION_BARS:
        raise ValueError("bar component leaves support")
    if not 0 <= remainder < M4E_TICKS_PER_SEMANTIC_BAR:
        raise ValueError("remainder component leaves support")
    return bars * M4E_TICKS_PER_SEMANTIC_BAR + remainder


@dataclass(frozen=True, slots=True)
class MelodyTarget:
    target_type: int
    time_mode: int = TIME_NONE
    time_bars: int = 0
    time_remainder: int = 0
    cardinality: int = 0
    anchor_mode: int = ANCHOR_NONE
    anchor_class: int = 0
    gaps: tuple[int, ...] = ()
    duration_bars: tuple[int, ...] = ()
    duration_remainders: tuple[int, ...] = ()

    @property
    def raw_note_count(self) -> int:
        return self.cardinality if self.target_type == TARGET_NOTE_GROUP else 0


def target_for_occurrence(
    stream: SparseMelodyStream,
    target_index: int,
    *,
    maximum_notes_per_group: int,
) -> MelodyTarget:
    """Encode one next occurrence without exposing it to the causal backbone."""

    target_index = int(target_index)
    if not 0 <= target_index <= len(stream.atoms):
        raise IndexError("target index leaves stream")
    if target_index == len(stream.atoms):
        return MelodyTarget(target_type=TARGET_EOS)

    atom = stream.atoms[target_index]
    onset = int(stream.anchors[target_index])
    previous = stream.atoms[target_index - 1] if target_index else None
    previous_onset = int(stream.anchors[target_index - 1]) if target_index else None

    if atom.is_rest:
        if previous is None or previous.is_rest:
            raise AssertionError("REST must follow a sounding NoteGroup")
        bars, remainder = split_ticks(atom.rest_ticks)
        return MelodyTarget(
            target_type=TARGET_REST,
            time_mode=TIME_REST_DURATION,
            time_bars=bars,
            time_remainder=remainder,
        )

    cardinality = len(atom.notes)
    if not 1 <= cardinality <= int(maximum_notes_per_group):
        raise ValueError("NoteGroup leaves frozen cardinality support")
    pitches = tuple(note.midi_pitch for note in atom.notes)
    low = pitches[0]
    if target_index == 0:
        time_mode = TIME_ABSOLUTE_START
        time_value = onset
        anchor_mode = ANCHOR_ABSOLUTE
        anchor_class = low
    elif previous is not None and previous.is_rest:
        if previous_onset is None:
            raise AssertionError("REST occurrence has no anchor")
        if onset != previous_onset + previous.rest_ticks:
            raise AssertionError("NoteGroup after REST is not fixed by REST end")
        time_mode = TIME_NONE
        time_value = 0
        anchor_mode = ANCHOR_ABSOLUTE
        anchor_class = low
    else:
        if previous is None or previous_onset is None or previous.is_rest:
            raise AssertionError("continuous NoteGroup predecessor is malformed")
        time_mode = TIME_ONSET_DELTA
        time_value = onset - previous_onset
        if time_value <= 0:
            raise AssertionError("NoteGroup onsets must be strictly ordered")
        previous_low = previous.notes[0].midi_pitch
        delta = low - previous_low
        if not -127 <= delta <= 127:
            raise AssertionError("pitch anchor delta leaves MIDI support")
        anchor_mode = ANCHOR_DELTA
        anchor_class = delta + PITCH_DELTA_OFFSET

    time_bars, time_remainder = split_ticks(time_value)
    gaps = tuple(right - left for left, right in zip(pitches, pitches[1:]))
    if any(not 0 <= gap < 128 for gap in gaps):
        raise AssertionError("ascending pitch gap leaves support")
    durations = tuple(split_ticks(note.duration_ticks) for note in atom.notes)
    return MelodyTarget(
        target_type=TARGET_NOTE_GROUP,
        time_mode=time_mode,
        time_bars=time_bars,
        time_remainder=time_remainder,
        cardinality=cardinality,
        anchor_mode=anchor_mode,
        anchor_class=anchor_class,
        gaps=gaps,
        duration_bars=tuple(value[0] for value in durations),
        duration_remainders=tuple(value[1] for value in durations),
    )


def exact_factorized_target_roundtrip(
    stream: SparseMelodyStream,
    *,
    maximum_notes_per_group: int,
) -> bool:
    """Decode all factorized targets and verify exact sparse-carrier recovery."""

    atoms: list[MelodyAtom] = []
    anchors: list[int] = []
    active_until: int | None = None
    for index in range(len(stream.atoms)):
        target = target_for_occurrence(
            stream, index, maximum_notes_per_group=maximum_notes_per_group
        )
        previous = atoms[-1] if atoms else None
        previous_onset = anchors[-1] if anchors else None
        if target.target_type == TARGET_REST:
            if active_until is None:
                return False
            duration = join_ticks(target.time_bars, target.time_remainder)
            atoms.append(MelodyAtom.rest(duration))
            anchors.append(active_until)
            continue
        if target.target_type != TARGET_NOTE_GROUP:
            return False

        if target.time_mode == TIME_ABSOLUTE_START:
            onset = join_ticks(target.time_bars, target.time_remainder)
        elif target.time_mode == TIME_ONSET_DELTA:
            if previous_onset is None:
                return False
            onset = previous_onset + join_ticks(
                target.time_bars, target.time_remainder
            )
        elif target.time_mode == TIME_NONE and previous is not None and previous.is_rest:
            if previous_onset is None:
                return False
            onset = previous_onset + previous.rest_ticks
        else:
            return False

        if target.anchor_mode == ANCHOR_ABSOLUTE:
            low = target.anchor_class
        elif target.anchor_mode == ANCHOR_DELTA:
            if previous is None or previous.is_rest:
                return False
            low = previous.notes[0].midi_pitch + (
                target.anchor_class - PITCH_DELTA_OFFSET
            )
        else:
            return False
        pitches = [low]
        for gap in target.gaps:
            pitches.append(pitches[-1] + int(gap))
        durations = [
            join_ticks(bars, remainder)
            for bars, remainder in zip(
                target.duration_bars, target.duration_remainders
            )
        ]
        if len(pitches) != target.cardinality or len(durations) != target.cardinality:
            return False
        note_group = MelodyAtom.note_group(
            SparseNote.from_midi(pitch, duration)
            for pitch, duration in zip(pitches, durations)
        )
        atoms.append(note_group)
        anchors.append(onset)
        group_end = max(onset + duration for duration in durations)
        active_until = group_end if active_until is None else max(active_until, group_end)

    eos = target_for_occurrence(
        stream, len(stream.atoms), maximum_notes_per_group=maximum_notes_per_group
    )
    return (
        eos.target_type == TARGET_EOS
        and tuple(atoms) == stream.atoms
        and tuple(anchors) == stream.anchors
    )


@dataclass(slots=True)
class MelodyCausalBatch:
    song_ids: tuple[str, ...]
    target_indices: tuple[int, ...]
    context_type: torch.Tensor
    context_valid: torch.Tensor
    context_lengths: torch.Tensor
    context_anchor_ticks: torch.Tensor
    context_absolute_bar: torch.Tensor
    context_beat: torch.Tensor
    context_fine_tick: torch.Tensor
    context_anchor_pitch: torch.Tensor
    context_cardinality: torch.Tensor
    context_pitches: torch.Tensor
    context_duration_bars: torch.Tensor
    context_duration_remainders: torch.Tensor
    context_note_mask: torch.Tensor
    context_rest_bars: torch.Tensor
    context_rest_remainders: torch.Tensor
    local_kind: torch.Tensor
    local_time_bars: torch.Tensor
    local_time_remainders: torch.Tensor
    local_pitch_relation: torch.Tensor
    target_type: torch.Tensor
    target_time_mode: torch.Tensor
    target_time_bars: torch.Tensor
    target_time_remainders: torch.Tensor
    target_cardinality: torch.Tensor
    target_anchor_mode: torch.Tensor
    target_anchor_class: torch.Tensor
    target_gaps: torch.Tensor
    target_gap_mask: torch.Tensor
    target_duration_bars: torch.Tensor
    target_duration_remainders: torch.Tensor
    target_note_mask: torch.Tensor
    raw_note_count: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.context_type.shape[0])

    @property
    def context_length(self) -> int:
        return int(self.context_type.shape[1])

    def to(self, device: torch.device | str) -> "MelodyCausalBatch":
        values = {
            field.name: value.to(device)
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), torch.Tensor)
        }
        return replace(self, **values)


def causal_example_indices(
    streams: Sequence[SparseMelodyStream],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (stream_index, target_index)
        for stream_index, stream in enumerate(streams)
        for target_index in range(len(stream.atoms) + 1)
    )


def _input_type(atom: MelodyAtom | None) -> int:
    if atom is None:
        return INPUT_BOS
    return INPUT_REST if atom.is_rest else INPUT_NOTE_GROUP


def _local_relation_kind(
    previous: MelodyAtom | None, current: MelodyAtom | None
) -> int:
    if previous is None or current is None:
        return LOCAL_START
    if not previous.is_rest and not current.is_rest:
        return LOCAL_NOTE_NOTE
    if not previous.is_rest and current.is_rest:
        return LOCAL_NOTE_REST
    if previous.is_rest and not current.is_rest:
        return LOCAL_REST_NOTE
    return LOCAL_REST_REST


def tensorize_causal_examples(
    streams: Sequence[SparseMelodyStream],
    examples: Sequence[tuple[int, int]],
    *,
    context_length: int,
    maximum_notes_per_group: int,
    context_start_indices: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> MelodyCausalBatch:
    if not examples:
        raise ValueError("cannot tensorize an empty causal batch")
    batch_size = len(examples)
    length = int(context_length)
    maximum_notes = int(maximum_notes_per_group)
    if length <= 0 or maximum_notes <= 0:
        raise ValueError("context and NoteGroup support must be positive")
    if context_start_indices is not None and len(context_start_indices) != batch_size:
        raise ValueError("context-start count differs from causal examples")

    sequence_shape = (batch_size, length)
    note_shape = (batch_size, length, maximum_notes)
    context_type = torch.full(sequence_shape, INPUT_PAD, dtype=torch.long)
    context_valid = torch.zeros(sequence_shape, dtype=torch.bool)
    context_lengths = torch.zeros(batch_size, dtype=torch.long)
    context_anchor_ticks = torch.zeros(sequence_shape, dtype=torch.long)
    context_absolute_bar = torch.zeros(sequence_shape, dtype=torch.long)
    context_beat = torch.zeros(sequence_shape, dtype=torch.long)
    context_fine_tick = torch.zeros(sequence_shape, dtype=torch.long)
    context_anchor_pitch = torch.zeros(sequence_shape, dtype=torch.long)
    context_cardinality = torch.zeros(sequence_shape, dtype=torch.long)
    context_pitches = torch.zeros(note_shape, dtype=torch.long)
    context_duration_bars = torch.zeros(note_shape, dtype=torch.long)
    context_duration_remainders = torch.zeros(note_shape, dtype=torch.long)
    context_note_mask = torch.zeros(note_shape, dtype=torch.bool)
    context_rest_bars = torch.zeros(sequence_shape, dtype=torch.long)
    context_rest_remainders = torch.zeros(sequence_shape, dtype=torch.long)
    local_kind = torch.zeros(sequence_shape, dtype=torch.long)
    local_time_bars = torch.zeros(sequence_shape, dtype=torch.long)
    local_time_remainders = torch.zeros(sequence_shape, dtype=torch.long)
    local_pitch_relation = torch.full(
        sequence_shape, PITCH_RELATION_NEUTRAL, dtype=torch.long
    )

    target_type = torch.zeros(batch_size, dtype=torch.long)
    target_time_mode = torch.zeros(batch_size, dtype=torch.long)
    target_time_bars = torch.zeros(batch_size, dtype=torch.long)
    target_time_remainders = torch.zeros(batch_size, dtype=torch.long)
    target_cardinality = torch.zeros(batch_size, dtype=torch.long)
    target_anchor_mode = torch.zeros(batch_size, dtype=torch.long)
    target_anchor_class = torch.zeros(batch_size, dtype=torch.long)
    target_gaps = torch.zeros((batch_size, maximum_notes - 1), dtype=torch.long)
    target_gap_mask = torch.zeros(
        (batch_size, maximum_notes - 1), dtype=torch.bool
    )
    target_duration_bars = torch.zeros(
        (batch_size, maximum_notes), dtype=torch.long
    )
    target_duration_remainders = torch.zeros_like(target_duration_bars)
    target_note_mask = torch.zeros(
        (batch_size, maximum_notes), dtype=torch.bool
    )
    raw_note_count = torch.zeros(batch_size, dtype=torch.long)
    song_ids: list[str] = []
    target_indices: list[int] = []

    for row, (stream_index, occurrence_index) in enumerate(examples):
        stream = streams[int(stream_index)]
        occurrence_index = int(occurrence_index)
        requested_start = (
            0
            if context_start_indices is None
            else int(context_start_indices[row])
        )
        if requested_start < 0 or requested_start > occurrence_index:
            raise ValueError("context start leaves the visible causal prefix")
        if (
            length == 1
            and context_start_indices is None
            and occurrence_index > 0
        ):
            # Target-only decoder views need just the immediately preceding
            # occurrence.  Avoid constructing and slicing the complete prefix
            # for every target in an all-position training batch.
            selected_atoms = [stream.atoms[occurrence_index - 1]]
            selected_anchors = [int(stream.anchors[occurrence_index - 1])]
        else:
            history_atoms: list[MelodyAtom | None] = [None]
            history_atoms.extend(stream.atoms[:occurrence_index])
            history_anchors = [0]
            history_anchors.extend(
                int(value) for value in stream.anchors[:occurrence_index]
            )
            # BOS is visible only when the requested physical history starts at
            # the beginning of the song. A left-truncated history begins with
            # its first real event, exactly as legacy fixed-unit truncation did.
            physical_start = 0 if requested_start == 0 else requested_start + 1
            start = max(physical_start, len(history_atoms) - length)
            selected_atoms = history_atoms[start:]
            selected_anchors = history_anchors[start:]
        context_lengths[row] = len(selected_atoms)
        song_ids.append(stream.song_id)
        target_indices.append(occurrence_index)

        for column, (atom, anchor) in enumerate(
            zip(selected_atoms, selected_anchors)
        ):
            context_valid[row, column] = True
            context_type[row, column] = _input_type(atom)
            context_anchor_ticks[row, column] = int(anchor)
            absolute_bar, local = divmod(
                int(anchor), M4E_TICKS_PER_SEMANTIC_BAR
            )
            if absolute_bar >= M4E_MAX_ABSOLUTE_BARS:
                raise ValueError("absolute melody position leaves support")
            context_absolute_bar[row, column] = absolute_bar
            context_beat[row, column] = local // M4E_TICKS_PER_BEAT
            context_fine_tick[row, column] = local % M4E_TICKS_PER_BEAT
            if atom is not None and atom.is_rest:
                bars, remainder = split_ticks(atom.rest_ticks)
                context_rest_bars[row, column] = bars
                context_rest_remainders[row, column] = remainder
            elif atom is not None:
                context_cardinality[row, column] = len(atom.notes)
                if len(atom.notes) > maximum_notes:
                    raise ValueError("input NoteGroup leaves support")
                context_anchor_pitch[row, column] = atom.notes[0].midi_pitch
                for note_index, note in enumerate(atom.notes):
                    bars, remainder = split_ticks(note.duration_ticks)
                    context_pitches[row, column, note_index] = note.midi_pitch
                    context_duration_bars[row, column, note_index] = bars
                    context_duration_remainders[row, column, note_index] = remainder
                    context_note_mask[row, column, note_index] = True

            previous_atom = selected_atoms[column - 1] if column else None
            previous_anchor = selected_anchors[column - 1] if column else None
            relation_kind = _local_relation_kind(previous_atom, atom)
            local_kind[row, column] = relation_kind
            if column and previous_anchor is not None:
                delta_time = int(anchor) - int(previous_anchor)
                if delta_time < 0:
                    raise AssertionError("visible musical time is not ordered")
                bars, remainder = split_ticks(delta_time)
                local_time_bars[row, column] = bars
                local_time_remainders[row, column] = remainder
            if relation_kind == LOCAL_NOTE_NOTE:
                if previous_atom is None or atom is None:
                    raise AssertionError("note relation is malformed")
                delta_pitch = atom.notes[0].midi_pitch - previous_atom.notes[0].midi_pitch
                local_pitch_relation[row, column] = (
                    delta_pitch + PITCH_DELTA_OFFSET
                )

        target = target_for_occurrence(
            stream,
            occurrence_index,
            maximum_notes_per_group=maximum_notes,
        )
        target_type[row] = target.target_type
        target_time_mode[row] = target.time_mode
        target_time_bars[row] = target.time_bars
        target_time_remainders[row] = target.time_remainder
        target_anchor_mode[row] = target.anchor_mode
        target_anchor_class[row] = target.anchor_class
        raw_note_count[row] = target.raw_note_count
        if target.target_type == TARGET_NOTE_GROUP:
            target_cardinality[row] = target.cardinality - 1
            for gap_index, gap in enumerate(target.gaps):
                target_gaps[row, gap_index] = gap
                target_gap_mask[row, gap_index] = True
            for note_index, (bars, remainder) in enumerate(
                zip(target.duration_bars, target.duration_remainders)
            ):
                target_duration_bars[row, note_index] = bars
                target_duration_remainders[row, note_index] = remainder
                target_note_mask[row, note_index] = True

    batch = MelodyCausalBatch(
        song_ids=tuple(song_ids),
        target_indices=tuple(target_indices),
        context_type=context_type,
        context_valid=context_valid,
        context_lengths=context_lengths,
        context_anchor_ticks=context_anchor_ticks,
        context_absolute_bar=context_absolute_bar,
        context_beat=context_beat,
        context_fine_tick=context_fine_tick,
        context_anchor_pitch=context_anchor_pitch,
        context_cardinality=context_cardinality,
        context_pitches=context_pitches,
        context_duration_bars=context_duration_bars,
        context_duration_remainders=context_duration_remainders,
        context_note_mask=context_note_mask,
        context_rest_bars=context_rest_bars,
        context_rest_remainders=context_rest_remainders,
        local_kind=local_kind,
        local_time_bars=local_time_bars,
        local_time_remainders=local_time_remainders,
        local_pitch_relation=local_pitch_relation,
        target_type=target_type,
        target_time_mode=target_time_mode,
        target_time_bars=target_time_bars,
        target_time_remainders=target_time_remainders,
        target_cardinality=target_cardinality,
        target_anchor_mode=target_anchor_mode,
        target_anchor_class=target_anchor_class,
        target_gaps=target_gaps,
        target_gap_mask=target_gap_mask,
        target_duration_bars=target_duration_bars,
        target_duration_remainders=target_duration_remainders,
        target_note_mask=target_note_mask,
        raw_note_count=raw_note_count,
    )
    return batch.to(device)


def signed_log_bucket(
    values: torch.Tensor,
    *,
    magnitude_buckets: int = 32,
    exact_magnitudes: int = 16,
    maximum_distance: int = 32 * M4E_TICKS_PER_SEMANTIC_BAR,
) -> torch.Tensor:
    """Map signed distances to zero plus symmetric exact/log buckets."""

    if not 0 < exact_magnitudes < magnitude_buckets:
        raise ValueError("signed bucket split is malformed")
    magnitude = values.abs().to(torch.long)
    small = (magnitude - 1).clamp(min=0, max=exact_magnitudes - 1)
    safe = magnitude.clamp_min(exact_magnitudes + 1).to(torch.float32)
    denominator = math.log(maximum_distance / float(exact_magnitudes))
    large = exact_magnitudes + torch.floor(
        torch.log(safe / float(exact_magnitudes))
        / max(denominator, 1e-9)
        * float(magnitude_buckets - exact_magnitudes)
    ).to(torch.long)
    magnitude_bucket = torch.where(
        magnitude <= exact_magnitudes, small, large
    ).clamp(max=magnitude_buckets - 1)
    positive = 1 + magnitude_bucket
    negative = 1 + magnitude_buckets + magnitude_bucket
    return torch.where(
        values == 0,
        torch.zeros_like(magnitude_bucket),
        torch.where(values > 0, positive, negative),
    )


class RelativeCausalBlock(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        feedforward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_width, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
        relative_bias: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, length, _ = values.shape
        if attention_mask is None:
            if relative_bias is None or relative_bias.shape != (
                batch_size,
                self.heads,
                length,
                length,
            ):
                raise ValueError("relative attention bias shape differs")
            attention_mask = prepare_relative_causal_attention_mask(
                relative_bias=relative_bias,
                valid=valid,
                dtype=values.dtype,
            )
        elif attention_mask.shape != (
            batch_size * self.heads,
            length,
            length,
        ):
            raise ValueError("prepared attention mask shape differs")
        normalized = self.norm_attention(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        values = values + attended
        values = values + self.feedforward(self.norm_feedforward(values))
        return values.masked_fill(~valid.unsqueeze(-1), 0.0)


def prepare_relative_causal_attention_mask(
    *,
    relative_bias: torch.Tensor,
    valid: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the shared dense mask once for all Transformer layers."""

    batch_size, heads, length, right = relative_bias.shape
    if right != length or valid.shape != (batch_size, length):
        raise ValueError("attention mask inputs differ in shape")
    causal = torch.triu(
        torch.ones((length, length), dtype=torch.bool, device=valid.device),
        diagonal=1,
    )
    mask = relative_bias.to(dtype=dtype).masked_fill(
        causal.view(1, 1, length, length), float("-inf")
    )
    return mask.masked_fill(
        ~valid[:, None, None, :], float("-inf")
    ).reshape(batch_size * heads, length, length)


class FactorizedNextUnitDecoder(nn.Module):
    def __init__(self, *, width: int, maximum_notes_per_group: int) -> None:
        super().__init__()
        self.maximum_notes_per_group = int(maximum_notes_per_group)
        self.normalization = nn.LayerNorm(width)
        self.condition_step = nn.GRUCell(width, width)

        self.type_head = nn.Linear(width, 3)
        self.type_embedding = nn.Embedding(3, width)
        self.time_mode_embedding = nn.Embedding(4, width)
        self.time_bar_head = nn.Linear(width, M4E_MAX_DURATION_BARS)
        self.time_bar_embedding = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.time_remainder_head = nn.Linear(width, M4E_TICKS_PER_SEMANTIC_BAR)
        self.time_remainder_embedding = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.cardinality_head = nn.Linear(width, maximum_notes_per_group)
        self.cardinality_embedding = nn.Embedding(maximum_notes_per_group, width)
        self.anchor_mode_embedding = nn.Embedding(3, width)
        self.anchor_head = nn.Linear(width, 255)
        self.anchor_embedding = nn.Embedding(255, width)
        self.gap_head = nn.Linear(width, 128)
        self.gap_embedding = nn.Embedding(128, width)
        self.duration_bar_head = nn.Linear(width, M4E_MAX_DURATION_BARS)
        self.duration_bar_embedding = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.duration_remainder_head = nn.Linear(
            width, M4E_TICKS_PER_SEMANTIC_BAR
        )
        self.duration_remainder_embedding = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )

    def _advance(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidate = self.condition_step(embedding, state)
        if mask is None:
            return candidate
        return torch.where(mask.unsqueeze(-1), candidate, state)

    @staticmethod
    def _categorical_nll(
        logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(logits, target, reduction="none")

    def losses(
        self,
        hidden: torch.Tensor,
        batch: MelodyCausalBatch,
        *,
        anchor_logit_bias: torch.Tensor | None = None,
        anchor_occurrence_embedding: torch.Tensor | None = None,
        anchor_occurrence_mask: torch.Tensor | None = None,
        factor_residuals: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        factor_residuals = {} if factor_residuals is None else factor_residuals
        unsupported = set(factor_residuals) - {"type", "time", "anchor"}
        if unsupported:
            raise ValueError(
                f"unsupported factor residuals: {sorted(unsupported)}"
            )
        for name, residual in factor_residuals.items():
            if residual.shape != hidden.shape:
                raise ValueError(f"{name} factor residual shape differs")

        def residual(name: str, state: torch.Tensor) -> torch.Tensor:
            value = factor_residuals.get(name)
            return state if value is None else state + value

        hidden = self.normalization(hidden)
        type_nll = self._categorical_nll(
            self.type_head(residual("type", hidden)), batch.target_type
        )
        state = self._advance(hidden, self.type_embedding(batch.target_type))
        state = self._advance(
            state, self.time_mode_embedding(batch.target_time_mode)
        )

        time_mask = batch.target_time_mode != TIME_NONE
        time_bar_nll = self._categorical_nll(
            self.time_bar_head(
                self.normalization(residual("time", state))
            ),
            batch.target_time_bars,
        ) * time_mask
        state = self._advance(
            state,
            self.time_bar_embedding(batch.target_time_bars),
            time_mask,
        )
        time_remainder_nll = self._categorical_nll(
            self.time_remainder_head(
                self.normalization(residual("time", state))
            ),
            batch.target_time_remainders,
        ) * time_mask
        state = self._advance(
            state,
            self.time_remainder_embedding(batch.target_time_remainders),
            time_mask,
        )
        time_nll = time_bar_nll + time_remainder_nll

        note_mask = batch.target_type == TARGET_NOTE_GROUP
        cardinality_nll = self._categorical_nll(
            self.cardinality_head(self.normalization(state)),
            batch.target_cardinality,
        ) * note_mask
        state = self._advance(
            state,
            self.cardinality_embedding(batch.target_cardinality),
            note_mask,
        )
        state = self._advance(
            state,
            self.anchor_mode_embedding(batch.target_anchor_mode),
            note_mask,
        )
        if anchor_occurrence_embedding is not None:
            if anchor_occurrence_embedding.shape != hidden.shape:
                raise ValueError("anchor occurrence embedding shape differs")
            occurrence_mask = (
                note_mask
                if anchor_occurrence_mask is None
                else anchor_occurrence_mask & note_mask
            )
            state = self._advance(
                state,
                anchor_occurrence_embedding,
                occurrence_mask,
            )
        anchor_logits = self.anchor_head(
            self.normalization(residual("anchor", state))
        )
        if anchor_logit_bias is not None:
            if anchor_logit_bias.shape != anchor_logits.shape:
                raise ValueError("anchor logit bias shape differs")
            anchor_logits = anchor_logits + anchor_logit_bias
        anchor_nll = self._categorical_nll(
            anchor_logits, batch.target_anchor_class
        ) * note_mask
        state = self._advance(
            state,
            self.anchor_embedding(batch.target_anchor_class),
            note_mask,
        )

        gap_nll = torch.zeros_like(type_nll)
        for slot in range(self.maximum_notes_per_group - 1):
            mask = batch.target_gap_mask[:, slot]
            current = self._categorical_nll(
                self.gap_head(self.normalization(state)),
                batch.target_gaps[:, slot],
            )
            gap_nll = gap_nll + current * mask
            state = self._advance(
                state,
                self.gap_embedding(batch.target_gaps[:, slot]),
                mask,
            )

        duration_nll = torch.zeros_like(type_nll)
        for slot in range(self.maximum_notes_per_group):
            mask = batch.target_note_mask[:, slot]
            bar_target = batch.target_duration_bars[:, slot]
            remainder_target = batch.target_duration_remainders[:, slot]
            bar_nll = self._categorical_nll(
                self.duration_bar_head(self.normalization(state)), bar_target
            )
            duration_nll = duration_nll + bar_nll * mask
            state = self._advance(
                state, self.duration_bar_embedding(bar_target), mask
            )
            remainder_nll = self._categorical_nll(
                self.duration_remainder_head(self.normalization(state)),
                remainder_target,
            )
            duration_nll = duration_nll + remainder_nll * mask
            state = self._advance(
                state,
                self.duration_remainder_embedding(remainder_target),
                mask,
            )

        total = (
            type_nll
            + time_nll
            + cardinality_nll
            + anchor_nll
            + gap_nll
            + duration_nll
        )
        return {
            "total": total,
            "type": type_nll,
            "time": time_nll,
            "cardinality": cardinality_nll,
            "anchor": anchor_nll,
            "gaps": gap_nll,
            "duration": duration_nll,
        }


class DualCoordinateMelodyModel(nn.Module):
    """One matched causal model with optional local/pairwise music relations."""

    def __init__(
        self,
        *,
        arm: str,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
    ) -> None:
        super().__init__()
        if arm not in M4L_ARMS:
            raise ValueError(f"unsupported M4L arm: {arm}")
        self.arm = arm
        self.use_temporal_relations = arm in (
            "A_TIME_ONLY",
            "A1_DUAL_COORDINATE",
        )
        self.use_pitch_relations = arm == "A1_DUAL_COORDINATE"
        self.width = int(width)
        self.heads = int(heads)
        self.maximum_notes_per_group = int(maximum_notes_per_group)

        self.input_type = nn.Embedding(4, width)
        self.pitch = nn.Embedding(128, width)
        self.duration_bars = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.duration_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.cardinality = nn.Embedding(maximum_notes_per_group + 1, width)
        self.rest_bars = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.rest_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.absolute_bar = nn.Embedding(M4E_MAX_ABSOLUTE_BARS, width)
        self.beat = nn.Embedding(M4E_BEATS_PER_SEMANTIC_BAR, width)
        self.fine_tick = nn.Embedding(M4E_TICKS_PER_BEAT, width)
        self.note_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.content_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.input_normalization = nn.LayerNorm(width)

        self.local_kind = nn.Embedding(6, width)
        self.local_time_bars = nn.Embedding(M4E_MAX_DURATION_BARS, width)
        self.local_time_remainders = nn.Embedding(
            M4E_TICKS_PER_SEMANTIC_BAR, width
        )
        self.local_pitch = nn.Embedding(PITCH_RELATION_CLASSES, width)
        self.time_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.pitch_bias = nn.Embedding(PITCH_RELATION_CLASSES, heads)
        self.meter_bias = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, heads)
        for embedding in (
            self.local_kind,
            self.local_time_bars,
            self.local_time_remainders,
            self.local_pitch,
            self.time_bias,
            self.pitch_bias,
            self.meter_bias,
        ):
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

    def encode_inputs(self, batch: MelodyCausalBatch) -> torch.Tensor:
        note_mask = batch.context_note_mask.unsqueeze(-1).to(torch.float32)
        notes = (
            self.pitch(batch.context_pitches)
            + self.duration_bars(batch.context_duration_bars)
            + self.duration_remainders(batch.context_duration_remainders)
        )
        notes = self.note_projection(notes)
        pooled = (notes * note_mask).sum(dim=2) / note_mask.sum(dim=2).clamp_min(1.0)
        pooled = pooled + self.cardinality(batch.context_cardinality)
        rest = self.rest_bars(batch.context_rest_bars) + self.rest_remainders(
            batch.context_rest_remainders
        )
        is_note = batch.context_type == INPUT_NOTE_GROUP
        is_rest = batch.context_type == INPUT_REST
        content = torch.zeros_like(pooled)
        content = torch.where(is_note.unsqueeze(-1), pooled, content)
        content = torch.where(is_rest.unsqueeze(-1), rest, content)
        content = self.content_projection(content)
        position = (
            self.absolute_bar(batch.context_absolute_bar)
            + self.beat(batch.context_beat)
            + self.fine_tick(batch.context_fine_tick)
        )
        values = self.input_type(batch.context_type) + content + position
        if self.use_temporal_relations:
            values = values + (
                self.local_kind(batch.local_kind)
                + self.local_time_bars(batch.local_time_bars)
                + self.local_time_remainders(batch.local_time_remainders)
            )
        if self.use_pitch_relations:
            values = values + self.local_pitch(batch.local_pitch_relation)
        values = self.input_normalization(values)
        return values.masked_fill(~batch.context_valid.unsqueeze(-1), 0.0)

    def relative_attention_bias(
        self, batch: MelodyCausalBatch
    ) -> torch.Tensor:
        shape = (
            batch.batch_size,
            self.heads,
            batch.context_length,
            batch.context_length,
        )
        if not self.use_temporal_relations:
            return torch.zeros(
                shape,
                dtype=self.input_type.weight.dtype,
                device=batch.context_type.device,
            )
        time_delta = (
            batch.context_anchor_ticks[:, :, None]
            - batch.context_anchor_ticks[:, None, :]
        )
        time_bucket = signed_log_bucket(time_delta)
        time = self.time_bias(time_bucket)
        meter_index = torch.remainder(
            time_delta, M4E_TICKS_PER_SEMANTIC_BAR
        )
        meter = self.meter_bias(meter_index)
        bias = time + meter
        if self.use_pitch_relations:
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
            bias = bias + self.pitch_bias(pitch_index)
        return bias.permute(0, 3, 1, 2)

    def causal_hidden(self, batch: MelodyCausalBatch) -> torch.Tensor:
        values = self.encode_inputs(batch)
        bias = self.relative_attention_bias(batch)
        for block in self.blocks:
            values = block(
                values, valid=batch.context_valid, relative_bias=bias
            )
        values = self.output_normalization(values)
        rows = torch.arange(batch.batch_size, device=values.device)
        return values[rows, batch.context_lengths - 1]

    def losses(self, batch: MelodyCausalBatch) -> dict[str, torch.Tensor]:
        return self.decoder.losses(self.causal_hidden(batch), batch)


def component_applicable_counts(batch: MelodyCausalBatch) -> dict[str, int]:
    note_count = int(batch.target_note_mask.sum().item())
    return {
        "total": batch.batch_size,
        "type": batch.batch_size,
        "time": int((batch.target_time_mode != TIME_NONE).sum().item()),
        "cardinality": int(
            (batch.target_type == TARGET_NOTE_GROUP).sum().item()
        ),
        "anchor": int((batch.target_type == TARGET_NOTE_GROUP).sum().item()),
        "gaps": int(batch.target_gap_mask.sum().item()),
        "duration": note_count,
        "raw_notes": int(batch.raw_note_count.sum().item()),
        "musical_units": int((batch.target_type != TARGET_EOS).sum().item()),
        "eos": int((batch.target_type == TARGET_EOS).sum().item()),
    }
