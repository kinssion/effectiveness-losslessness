from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from .dual_coordinate_melody import (
    INPUT_NOTE_GROUP,
    DualCoordinateMelodyModel,
    MelodyCausalBatch,
    tensorize_causal_examples,
)
from .harmonic_conditioning_melody import ChordConditionBatch
from .harmonic_pitch_energy import CandidateWiseHarmonicEnergy
from .hierarchical_tonic_melody import GlobalKey
from .sparse_melody_bpe import SparseMelodyStream


M4L_PHASE4A_SCHEMA = "usmm.m4l.absolute_tonic_relative.phase4a.v1"
PHASE4A_A0 = "A0_ABSOLUTE_FULL_A1"
PHASE4A_A1 = "A1_ABSOLUTE_PLUS_TONIC_RELATIVE"
PHASE4A_ARMS = (PHASE4A_A0, PHASE4A_A1)
RELATIVE_OCTAVE_CLASSES = 12


@dataclass(slots=True)
class TonicCoordinateBatch:
    melody: MelodyCausalBatch
    tonic: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.melody.batch_size

    def to(self, device: torch.device | str) -> "TonicCoordinateBatch":
        return TonicCoordinateBatch(
            melody=self.melody.to(device),
            tonic=self.tonic.to(device),
        )


def tensorize_tonic_coordinate_examples(
    streams: Sequence[SparseMelodyStream],
    examples: Sequence[tuple[int, int]],
    keys: Mapping[str, GlobalKey],
    *,
    context_length: int,
    maximum_notes_per_group: int,
    context_start_indices: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> TonicCoordinateBatch:
    melody = tensorize_causal_examples(
        streams,
        examples,
        context_length=context_length,
        maximum_notes_per_group=maximum_notes_per_group,
        context_start_indices=context_start_indices,
        device="cpu",
    )
    tonic = torch.tensor(
        [
            int(keys[streams[int(stream_index)].song_id].tonic)
            for stream_index, _ in examples
        ],
        dtype=torch.long,
    )
    return TonicCoordinateBatch(melody=melody, tonic=tonic).to(device)


class AbsoluteTonicRelativeMelodyModel(DualCoordinateMelodyModel):
    """Full-A1 melody model with an optional oracle-tonic residual coordinate."""

    def __init__(
        self,
        *,
        phase4a_arm: str,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_notes_per_group: int = 8,
    ) -> None:
        if phase4a_arm not in PHASE4A_ARMS:
            raise ValueError(f"unsupported Phase 4A arm: {phase4a_arm}")
        super().__init__(
            arm="A1_DUAL_COORDINATE",
            width=width,
            heads=heads,
            layers=layers,
            feedforward_width=feedforward_width,
            dropout=dropout,
            maximum_notes_per_group=maximum_notes_per_group,
        )
        self.phase4a_arm = phase4a_arm
        self.use_tonic_coordinate = phase4a_arm == PHASE4A_A1
        if self.use_tonic_coordinate:
            self.tonic_relative_pc = nn.Embedding(12, width)
            self.tonic_relative_octave = nn.Embedding(
                RELATIVE_OCTAVE_CLASSES, width
            )
            self.tonic_coordinate_projection = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
            )
            self.tonic_coordinate_normalization = nn.LayerNorm(width)
            self.tonic_coordinate_scale = nn.Parameter(torch.zeros(()))

    def tonic_relative_note_residual(
        self, batch: TonicCoordinateBatch
    ) -> torch.Tensor:
        if not self.use_tonic_coordinate:
            return torch.zeros(
                (
                    batch.batch_size,
                    batch.melody.context_length,
                    self.width,
                ),
                dtype=self.input_type.weight.dtype,
                device=batch.melody.context_type.device,
            )
        melody = batch.melody
        relative_pitch = melody.context_pitches - batch.tonic[:, None, None]
        relative_pc = torch.remainder(relative_pitch, 12)
        relative_octave = (
            torch.div(relative_pitch, 12, rounding_mode="floor") + 1
        )
        invalid = melody.context_note_mask & (
            (relative_octave < 0)
            | (relative_octave >= RELATIVE_OCTAVE_CLASSES)
        )
        if bool(invalid.any()):
            raise ValueError("tonic-relative octave leaves MIDI support")
        relative_octave = relative_octave.clamp(
            min=0, max=RELATIVE_OCTAVE_CLASSES - 1
        )
        per_note = self.tonic_coordinate_projection(
            self.tonic_relative_pc(relative_pc)
            + self.tonic_relative_octave(relative_octave)
        )
        note_mask = melody.context_note_mask.unsqueeze(-1).to(per_note.dtype)
        pooled = (per_note * note_mask).sum(dim=2) / note_mask.sum(
            dim=2
        ).clamp_min(1.0)
        note_occurrence = melody.context_type == INPUT_NOTE_GROUP
        pooled = pooled.masked_fill(~note_occurrence.unsqueeze(-1), 0.0)
        pooled = self.tonic_coordinate_normalization(pooled)
        return pooled.masked_fill(
            ~melody.context_valid.unsqueeze(-1), 0.0
        )

    def encode_tonic_inputs(
        self, batch: TonicCoordinateBatch
    ) -> torch.Tensor:
        absolute = super().encode_inputs(batch.melody)
        if not self.use_tonic_coordinate:
            return absolute
        return (
            absolute
            + self.tonic_coordinate_scale
            * self.tonic_relative_note_residual(batch)
        ).masked_fill(~batch.melody.context_valid.unsqueeze(-1), 0.0)

    def causal_hidden(
        self, batch: TonicCoordinateBatch
    ) -> torch.Tensor:
        melody = batch.melody
        values = self.encode_tonic_inputs(batch)
        bias = self.relative_attention_bias(melody)
        for block in self.blocks:
            values = block(
                values, valid=melody.context_valid, relative_bias=bias
            )
        values = self.output_normalization(values)
        rows = torch.arange(batch.batch_size, device=values.device)
        return values[rows, melody.context_lengths - 1]

    def losses(
        self, batch: TonicCoordinateBatch
    ) -> dict[str, torch.Tensor]:
        return self.decoder.losses(
            self.causal_hidden(batch), batch.melody
        )


class FrozenCoordinateHarmonicEnergyModel(nn.Module):
    """Frozen Phase 4A melody prior plus the established pitch-only energy."""

    def __init__(
        self,
        *,
        base: AbsoluteTonicRelativeMelodyModel,
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
