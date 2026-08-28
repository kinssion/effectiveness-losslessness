from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_DURATION_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .dual_coordinate_melody import (
    ANCHOR_ABSOLUTE,
    ANCHOR_DELTA,
    INPUT_BOS,
    INPUT_NOTE_GROUP,
    INPUT_PAD,
    INPUT_REST,
    PITCH_DELTA_OFFSET,
    PITCH_RELATION_NEUTRAL,
    TARGET_NOTE_GROUP,
    DualCoordinateMelodyModel,
    MelodyCausalBatch,
    signed_log_bucket,
    tensorize_causal_examples,
)
from .harmonic_conditioning_melody import (
    CHORD_SLOT_TICKS,
    NO_CHORD_ROOT,
    ChordConditionBatch,
)
from .sparse_melody_bpe import SparseMelodyStream


M4L_PHASE3F_SCHEMA = "usmm.m4l.tonic_hierarchical_forward.phase3f.v1"
MODE_MAJOR = 0
MODE_MINOR = 1
MODE_CLASSES = 2
RELATIVE_OCTAVE_CLASSES = 12
ANCHOR_CLASSES = 255

_PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}


@dataclass(frozen=True, slots=True)
class GlobalKey:
    tonic: int
    mode: int
    source_label: str

    def __post_init__(self) -> None:
        if not 0 <= int(self.tonic) < 12:
            raise ValueError("global tonic leaves pitch-class support")
        if int(self.mode) not in (MODE_MAJOR, MODE_MINOR):
            raise ValueError("global mode must be major or minor")


def parse_global_key_label(label: str) -> GlobalKey:
    root, separator, mode = str(label).strip().partition(":")
    if not separator:
        raise ValueError(f"malformed POP909 global key: {label}")
    root_key = root.strip().upper()
    normalized_mode = mode.strip().lower()
    if root_key not in _PITCH_CLASS:
        raise ValueError(f"unsupported POP909 tonic: {root}")
    if normalized_mode in {"maj", "major"}:
        mode_id = MODE_MAJOR
    elif normalized_mode in {"min", "minor"}:
        mode_id = MODE_MINOR
    else:
        raise ValueError(f"unsupported POP909 global mode: {mode}")
    return GlobalKey(
        tonic=_PITCH_CLASS[root_key],
        mode=mode_id,
        source_label=f"{root}:{'maj' if mode_id == MODE_MAJOR else 'min'}",
    )


def read_global_key(path: Path) -> GlobalKey:
    """Read one song-level key by choosing the longest annotated interval."""

    candidates: list[tuple[float, GlobalKey]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        start = float(fields[0])
        end = float(fields[1])
        if end <= start:
            continue
        candidates.append((end - start, parse_global_key_label(fields[2])))
    if not candidates:
        raise ValueError(f"no valid global key annotation: {path}")
    candidates.sort(key=lambda value: value[0], reverse=True)
    return candidates[0][1]


@dataclass(slots=True)
class TonicHierarchicalBatch:
    melody: MelodyCausalBatch
    tonic: torch.Tensor
    mode: torch.Tensor
    target_anchor_ticks: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.melody.batch_size

    def to(self, device: torch.device | str) -> "TonicHierarchicalBatch":
        return TonicHierarchicalBatch(
            melody=self.melody.to(device),
            tonic=self.tonic.to(device),
            mode=self.mode.to(device),
            target_anchor_ticks=self.target_anchor_ticks.to(device),
        )


def tensorize_tonic_hierarchical_examples(
    streams: Sequence[SparseMelodyStream],
    examples: Sequence[tuple[int, int]],
    keys: Mapping[str, GlobalKey],
    *,
    context_length: int,
    maximum_notes_per_group: int,
    device: torch.device | str = "cpu",
) -> TonicHierarchicalBatch:
    melody = tensorize_causal_examples(
        streams,
        examples,
        context_length=context_length,
        maximum_notes_per_group=maximum_notes_per_group,
        device="cpu",
    )
    tonics: list[int] = []
    modes: list[int] = []
    targets: list[int] = []
    for stream_index, target_index in examples:
        stream = streams[int(stream_index)]
        key = keys[stream.song_id]
        tonics.append(int(key.tonic))
        modes.append(int(key.mode))
        target_index = int(target_index)
        if target_index < len(stream.anchors):
            target_tick = int(stream.anchors[target_index])
        elif stream.anchors:
            target_tick = int(stream.anchors[-1])
        else:
            target_tick = 0
        targets.append(target_tick)
    return TonicHierarchicalBatch(
        melody=melody,
        tonic=torch.tensor(tonics, dtype=torch.long),
        mode=torch.tensor(modes, dtype=torch.long),
        target_anchor_ticks=torch.tensor(targets, dtype=torch.long),
    ).to(device)


class TonicChordPitchEnergy(nn.Module):
    """Learned key/chord soft constraint over candidate anchor pitch classes."""

    def __init__(
        self, *, width: int, chord_quality_classes: int
    ) -> None:
        super().__init__()
        self.key_degree_energy = nn.Parameter(torch.zeros(MODE_CLASSES, 12))
        self.chord_relative_energy = nn.Parameter(
            torch.zeros(chord_quality_classes, 12)
        )
        self.hidden_normalization = nn.LayerNorm(width)
        self.hidden_gate = nn.Linear(width, 1, bias=False)
        self.beat_gate = nn.Embedding(M4E_BEATS_PER_SEMANTIC_BAR, 1)
        self.fine_gate = nn.Embedding(M4E_TICKS_PER_BEAT, 1)
        self.gate_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.hidden_gate.weight)
        nn.init.zeros_(self.beat_gate.weight)
        nn.init.zeros_(self.fine_gate.weight)

    @staticmethod
    def _current_chord(
        chord: ChordConditionBatch, target_ticks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = target_ticks[:, None] - chord.anchor_ticks
        selected = chord.valid & (delta >= 0) & (delta < CHORD_SLOT_TICKS)
        if bool((selected.sum(dim=-1) != 1).any()):
            raise ValueError("target does not map to exactly one chord slot")
        index = selected.to(torch.long).argmax(dim=-1)
        rows = torch.arange(target_ticks.shape[0], device=target_ticks.device)
        return chord.root_classes[rows, index], chord.quality_ids[rows, index]

    def forward(
        self,
        hidden: torch.Tensor,
        batch: TonicHierarchicalBatch,
        chord: ChordConditionBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        melody = batch.melody
        root, quality = self._current_chord(
            chord, batch.target_anchor_ticks
        )
        rows = torch.arange(batch.batch_size, device=hidden.device)
        last = melody.context_lengths - 1
        previous_pitch = melody.context_anchor_pitch[rows, last]
        classes = torch.arange(ANCHOR_CLASSES, device=hidden.device)[None, :]
        absolute_candidate = classes.expand(batch.batch_size, -1)
        delta_candidate = previous_pitch[:, None] + classes - PITCH_DELTA_OFFSET
        absolute_mode = melody.target_anchor_mode == ANCHOR_ABSOLUTE
        delta_mode = melody.target_anchor_mode == ANCHOR_DELTA
        candidate_pitch = torch.where(
            absolute_mode[:, None], absolute_candidate, delta_candidate
        )
        valid_candidate = (
            (absolute_mode[:, None] & (classes < 128))
            | (
                delta_mode[:, None]
                & (delta_candidate >= 0)
                & (delta_candidate < 128)
            )
        )
        relative_to_tonic = torch.remainder(
            candidate_pitch - batch.tonic[:, None], 12
        ).to(torch.long)
        key_energy = self.key_degree_energy[
            batch.mode[:, None].expand_as(relative_to_tonic),
            relative_to_tonic,
        ]

        safe_root = root.clamp(min=0, max=11)
        relative_to_chord = torch.remainder(
            candidate_pitch - safe_root[:, None], 12
        ).to(torch.long)
        chord_energy = self.chord_relative_energy[
            quality[:, None].expand_as(relative_to_chord),
            relative_to_chord,
        ]
        chord_energy = torch.where(
            root[:, None] == NO_CHORD_ROOT,
            torch.zeros_like(chord_energy),
            chord_energy,
        )

        beat = torch.remainder(
            batch.target_anchor_ticks, M4E_TICKS_PER_SEMANTIC_BAR
        ) // M4E_TICKS_PER_BEAT
        fine = torch.remainder(
            batch.target_anchor_ticks, M4E_TICKS_PER_BEAT
        )
        gate = torch.sigmoid(
            self.hidden_gate(self.hidden_normalization(hidden)).squeeze(-1)
            + self.beat_gate(beat).squeeze(-1)
            + self.fine_gate(fine).squeeze(-1)
            + self.gate_bias
        )
        note_row = melody.target_type == TARGET_NOTE_GROUP
        usable = valid_candidate & note_row[:, None]
        bias = (gate[:, None] * (key_energy + chord_energy)).masked_fill(
            ~usable, 0.0
        )
        return bias, {
            "gate": gate,
            "target_anchor_ticks": batch.target_anchor_ticks,
            "tonic": batch.tonic,
            "mode": batch.mode,
            "chord_root": root,
            "chord_quality": quality,
            "usable_candidate_count": usable.sum(dim=-1),
        }


class TonicRelativeHierarchicalMelodyModel(DualCoordinateMelodyModel):
    """Tonic-relative Micro-M4L plus causal continuous bar-prefix hierarchy."""

    def __init__(
        self,
        *,
        chord_quality_classes: int,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
    ) -> None:
        super().__init__(
            arm="A1_DUAL_COORDINATE",
            width=width,
            heads=heads,
            layers=layers,
            feedforward_width=feedforward_width,
            dropout=dropout,
            maximum_notes_per_group=maximum_notes_per_group,
        )
        del self.pitch
        self.relative_pitch_class = nn.Embedding(12, width)
        self.relative_octave = nn.Embedding(RELATIVE_OCTAVE_CLASSES, width)
        self.global_mode = nn.Embedding(MODE_CLASSES, width)
        self.tonic_output = nn.Embedding(12, width)

        self.macro_bos = nn.Parameter(torch.zeros(width))
        self.macro_gru = nn.GRU(width, width, batch_first=True)
        self.macro_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU()
        )
        self.macro_prefix_normalization = nn.LayerNorm(width)
        self.harmonic_energy = TonicChordPitchEnergy(
            width=width, chord_quality_classes=chord_quality_classes
        )

    def encode_tonic_inputs(
        self, batch: TonicHierarchicalBatch
    ) -> torch.Tensor:
        melody = batch.melody
        note_mask = melody.context_note_mask.unsqueeze(-1).to(torch.float32)
        relative_pitch = melody.context_pitches - batch.tonic[:, None, None]
        relative_pc = torch.remainder(relative_pitch, 12)
        relative_octave = torch.div(
            relative_pitch, 12, rounding_mode="floor"
        ) + 1
        if bool(
            (
                melody.context_note_mask
                & (
                    (relative_octave < 0)
                    | (relative_octave >= RELATIVE_OCTAVE_CLASSES)
                )
            ).any()
        ):
            raise ValueError("tonic-relative octave leaves MIDI support")
        relative_octave = relative_octave.clamp(
            min=0, max=RELATIVE_OCTAVE_CLASSES - 1
        )
        notes = (
            self.relative_pitch_class(relative_pc)
            + self.relative_octave(relative_octave)
            + self.duration_bars(melody.context_duration_bars)
            + self.duration_remainders(melody.context_duration_remainders)
        )
        notes = self.note_projection(notes)
        pooled = (notes * note_mask).sum(dim=2) / note_mask.sum(dim=2).clamp_min(1.0)
        pooled = pooled + self.cardinality(melody.context_cardinality)
        rest = self.rest_bars(melody.context_rest_bars) + self.rest_remainders(
            melody.context_rest_remainders
        )
        is_note = melody.context_type == INPUT_NOTE_GROUP
        is_rest = melody.context_type == INPUT_REST
        content = torch.zeros_like(pooled)
        content = torch.where(is_note.unsqueeze(-1), pooled, content)
        content = torch.where(is_rest.unsqueeze(-1), rest, content)
        content = self.content_projection(content)
        position = (
            self.absolute_bar(melody.context_absolute_bar)
            + self.beat(melody.context_beat)
            + self.fine_tick(melody.context_fine_tick)
        )
        values = (
            self.input_type(melody.context_type)
            + content
            + position
            + self.global_mode(batch.mode)[:, None, :]
            + self.local_kind(melody.local_kind)
            + self.local_time_bars(melody.local_time_bars)
            + self.local_time_remainders(melody.local_time_remainders)
            + self.local_pitch(melody.local_pitch_relation)
        )
        values = self.input_normalization(values)
        return values.masked_fill(~melody.context_valid.unsqueeze(-1), 0.0)

    def _relative_bias_from_coordinates(
        self,
        *,
        anchor_ticks: torch.Tensor,
        anchor_pitch: torch.Tensor,
        context_type: torch.Tensor,
    ) -> torch.Tensor:
        time_delta = anchor_ticks[:, :, None] - anchor_ticks[:, None, :]
        bias = self.time_bias(signed_log_bucket(time_delta))
        meter_index = torch.remainder(
            time_delta, M4E_TICKS_PER_SEMANTIC_BAR
        )
        bias = bias + self.meter_bias(meter_index)
        pitch_delta = (
            anchor_pitch[:, :, None] - anchor_pitch[:, None, :]
        ).clamp(min=-127, max=127)
        pitched = (
            (context_type[:, :, None] == INPUT_NOTE_GROUP)
            & (context_type[:, None, :] == INPUT_NOTE_GROUP)
        )
        pitch_index = torch.where(
            pitched,
            pitch_delta + PITCH_DELTA_OFFSET,
            torch.full_like(pitch_delta, PITCH_RELATION_NEUTRAL),
        )
        bias = bias + self.pitch_bias(pitch_index)
        return bias.permute(0, 3, 1, 2)

    def _micro_sequence(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
        anchor_ticks: torch.Tensor,
        anchor_pitch: torch.Tensor,
        context_type: torch.Tensor,
    ) -> torch.Tensor:
        bias = self._relative_bias_from_coordinates(
            anchor_ticks=anchor_ticks,
            anchor_pitch=anchor_pitch,
            context_type=context_type,
        )
        for block in self.blocks:
            values = block(values, valid=valid, relative_bias=bias)
        return self.output_normalization(values).masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )

    def bottom_up_sequence(
        self, batch: TonicHierarchicalBatch
    ) -> torch.Tensor:
        melody = batch.melody
        return self._micro_sequence(
            self.encode_tonic_inputs(batch),
            valid=melody.context_valid,
            anchor_ticks=melody.context_anchor_ticks,
            anchor_pitch=melody.context_anchor_pitch,
            context_type=melody.context_type,
        )

    def macro_prefix(
        self,
        batch: TonicHierarchicalBatch,
        bottom_up: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        melody = batch.melody
        bottom_up = self.bottom_up_sequence(batch) if bottom_up is None else bottom_up
        target_bars = torch.div(
            batch.target_anchor_ticks,
            M4E_TICKS_PER_SEMANTIC_BAR,
            rounding_mode="floor",
        )
        sounding_timeline = (
            melody.context_valid & (melody.context_type != INPUT_BOS)
        )
        eligible = sounding_timeline & (
            melody.context_absolute_bar < target_bars[:, None]
        )
        maximum_bar = max(int(target_bars.max().item()), 1)
        bar_sums = bottom_up.new_zeros(
            batch.batch_size, maximum_bar, bottom_up.shape[-1]
        )
        bar_counts = bottom_up.new_zeros(batch.batch_size, maximum_bar)
        safe_bars = melody.context_absolute_bar.clamp(
            min=0, max=maximum_bar - 1
        )
        bar_sums.scatter_add_(
            1,
            safe_bars[:, :, None].expand(-1, -1, bottom_up.shape[-1]),
            bottom_up * eligible[:, :, None],
        )
        bar_counts.scatter_add_(
            1, safe_bars, eligible.to(bottom_up.dtype)
        )
        present = bar_counts > 0
        sequences: list[torch.Tensor] = []
        completed_counts: list[int] = []
        for row in range(batch.batch_size):
            selected = present[row]
            summaries = bar_sums[row, selected] / bar_counts[
                row, selected, None
            ].clamp_min(1.0)
            bos = self.macro_bos + self.global_mode(batch.mode[row])
            sequences.append(torch.cat((bos[None, :], summaries), dim=0))
            completed_counts.append(int(selected.sum().item()))
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            dtype=torch.long,
            device=bottom_up.device,
        )
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, final = self.macro_gru(packed)
        prefix = self.macro_projection(final[-1])
        return prefix, {
            "completed_bar_counts": torch.tensor(
                completed_counts, dtype=torch.long, device=bottom_up.device
            ),
            "macro_sequence_lengths": lengths,
        }

    def _insert_macro_prefix(
        self,
        batch: TonicHierarchicalBatch,
        values: torch.Tensor,
        prefix: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        melody = batch.melody
        value_rows: list[torch.Tensor] = []
        tick_rows: list[torch.Tensor] = []
        pitch_rows: list[torch.Tensor] = []
        type_rows: list[torch.Tensor] = []
        insertion_indices: list[int] = []
        lengths: list[int] = []
        for row in range(batch.batch_size):
            length = int(melody.context_lengths[row].item())
            target_bar = int(
                batch.target_anchor_ticks[row].item()
                // M4E_TICKS_PER_SEMANTIC_BAR
            )
            types = melody.context_type[row, :length]
            bars = melody.context_absolute_bar[row, :length]
            current = torch.nonzero(
                (bars >= target_bar) & (types != INPUT_BOS), as_tuple=False
            ).flatten()
            insertion = int(current[0].item()) if current.numel() else length
            insertion_indices.append(insertion)
            lengths.append(length + 1)
            value_rows.append(
                torch.cat(
                    (
                        values[row, :insertion],
                        self.macro_prefix_normalization(prefix[row])[None, :],
                        values[row, insertion:length],
                    ),
                    dim=0,
                )
            )
            prefix_tick = torch.tensor(
                [target_bar * M4E_TICKS_PER_SEMANTIC_BAR],
                dtype=melody.context_anchor_ticks.dtype,
                device=values.device,
            )
            tick_rows.append(
                torch.cat(
                    (
                        melody.context_anchor_ticks[row, :insertion],
                        prefix_tick,
                        melody.context_anchor_ticks[row, insertion:length],
                    )
                )
            )
            neutral = torch.zeros(1, dtype=torch.long, device=values.device)
            pitch_rows.append(
                torch.cat(
                    (
                        melody.context_anchor_pitch[row, :insertion],
                        neutral,
                        melody.context_anchor_pitch[row, insertion:length],
                    )
                )
            )
            type_rows.append(
                torch.cat(
                    (
                        types[:insertion],
                        torch.full_like(neutral, INPUT_PAD),
                        types[insertion:length],
                    )
                )
            )
        padded_values = pad_sequence(value_rows, batch_first=True)
        padded_ticks = pad_sequence(tick_rows, batch_first=True)
        padded_pitches = pad_sequence(pitch_rows, batch_first=True)
        padded_types = pad_sequence(
            type_rows, batch_first=True, padding_value=INPUT_PAD
        )
        lengths_tensor = torch.tensor(
            lengths, dtype=torch.long, device=values.device
        )
        valid = (
            torch.arange(padded_values.shape[1], device=values.device)[None, :]
            < lengths_tensor[:, None]
        )
        return (
            padded_values,
            valid,
            padded_ticks,
            padded_pitches,
            padded_types,
            torch.tensor(insertion_indices, dtype=torch.long, device=values.device),
        )

    def hierarchical_hidden(
        self, batch: TonicHierarchicalBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bottom_up = self.bottom_up_sequence(batch)
        prefix, macro_receipt = self.macro_prefix(batch, bottom_up)
        encoded = self.encode_tonic_inputs(batch)
        (
            augmented,
            valid,
            ticks,
            pitches,
            types,
            insertion_indices,
        ) = self._insert_macro_prefix(batch, encoded, prefix)
        values = self._micro_sequence(
            augmented,
            valid=valid,
            anchor_ticks=ticks,
            anchor_pitch=pitches,
            context_type=types,
        )
        rows = torch.arange(batch.batch_size, device=values.device)
        lengths = valid.sum(dim=-1)
        hidden = values[rows, lengths - 1]
        return hidden, {
            **macro_receipt,
            "insertion_indices": insertion_indices,
            "macro_prefix": prefix,
        }

    def losses_with_chords(
        self,
        batch: TonicHierarchicalBatch,
        chord: ChordConditionBatch,
    ) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
        hidden, hierarchy = self.hierarchical_hidden(batch)
        harmonic_bias, harmonic = self.harmonic_energy(hidden, batch, chord)
        absolute_anchor = (
            batch.melody.target_anchor_mode == ANCHOR_ABSOLUTE
        )
        losses = self.decoder.losses(
            hidden,
            batch.melody,
            anchor_logit_bias=harmonic_bias,
            anchor_occurrence_embedding=self.tonic_output(batch.tonic),
            anchor_occurrence_mask=absolute_anchor,
        )
        return losses, {**hierarchy, **harmonic}

    def losses(self, batch: TonicHierarchicalBatch) -> Mapping[str, torch.Tensor]:
        raise RuntimeError("Phase 3F requires the local chord condition stream")
