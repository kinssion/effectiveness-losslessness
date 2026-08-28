from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from unified_structured_music.dual_coordinate_melody import (
    ANCHOR_ABSOLUTE,
    ANCHOR_DELTA,
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
    PITCH_DELTA_OFFSET,
    TARGET_NOTE_GROUP,
    TIME_ABSOLUTE_START,
    TIME_NONE,
    TIME_ONSET_DELTA,
    MelodyCausalBatch,
)
from unified_structured_music.harmonic_conditioning_melody import (
    CHORD_SLOT_TICKS,
    NO_CHORD_ROOT,
    ChordConditionBatch,
    HarmonicConditionedMelodyModel,
)


M4L_PHASE3C_SCHEMA = "usmm.m4l.harmonic_pitch_energy.phase3c.v1"
ANCHOR_CLASSES = 255


def target_anchor_ticks_from_batch(batch: MelodyCausalBatch) -> torch.Tensor:
    rows = torch.arange(batch.batch_size, device=batch.context_type.device)
    last_index = batch.context_lengths - 1
    previous_anchor = batch.context_anchor_ticks[rows, last_index]
    explicit = (
        batch.target_time_bars * M4E_TICKS_PER_SEMANTIC_BAR
        + batch.target_time_remainders
    )
    previous_rest = (
        batch.context_rest_bars[rows, last_index] * M4E_TICKS_PER_SEMANTIC_BAR
        + batch.context_rest_remainders[rows, last_index]
    )
    target = previous_anchor
    target = torch.where(
        batch.target_time_mode == TIME_ABSOLUTE_START, explicit, target
    )
    target = torch.where(
        batch.target_time_mode == TIME_ONSET_DELTA,
        previous_anchor + explicit,
        target,
    )
    target = torch.where(
        batch.target_time_mode == TIME_NONE,
        previous_anchor + previous_rest,
        target,
    )
    return target


def current_chord_at_target(
    chord: ChordConditionBatch, target_anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = target_anchors[:, None] - chord.anchor_ticks
    selected = (
        chord.valid & (delta >= 0) & (delta < CHORD_SLOT_TICKS)
    )
    if bool((selected.sum(dim=-1) != 1).any()):
        raise ValueError("target does not map to exactly one coarse chord slot")
    index = selected.to(torch.long).argmax(dim=-1)
    rows = torch.arange(target_anchors.shape[0], device=target_anchors.device)
    return chord.root_classes[rows, index], chord.quality_ids[rows, index]


class CandidateWiseHarmonicEnergy(nn.Module):
    """A learned pitch-class energy field with no chord-tone teacher."""

    def __init__(self, *, width: int, chord_quality_classes: int) -> None:
        super().__init__()
        self.quality_relative_pc_energy = nn.Parameter(
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

    def forward(
        self,
        hidden: torch.Tensor,
        batch: MelodyCausalBatch,
        chord: ChordConditionBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_anchors = target_anchor_ticks_from_batch(batch)
        chord_root, chord_quality = current_chord_at_target(
            chord, target_anchors
        )
        rows = torch.arange(batch.batch_size, device=hidden.device)
        last_index = batch.context_lengths - 1
        previous_pitch = batch.context_anchor_pitch[rows, last_index]
        classes = torch.arange(ANCHOR_CLASSES, device=hidden.device)[None, :]
        absolute_candidate = classes.expand(batch.batch_size, -1)
        delta_candidate = previous_pitch[:, None] + classes - PITCH_DELTA_OFFSET
        absolute_mode = batch.target_anchor_mode == ANCHOR_ABSOLUTE
        delta_mode = batch.target_anchor_mode == ANCHOR_DELTA
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
        relative_pc = torch.remainder(
            candidate_pitch - chord_root[:, None], 12
        ).to(torch.long)
        quality_index = chord_quality[:, None].expand_as(relative_pc)
        energy = self.quality_relative_pc_energy[quality_index, relative_pc]

        beat = torch.remainder(
            target_anchors, M4E_TICKS_PER_SEMANTIC_BAR
        ) // M4E_TICKS_PER_BEAT
        fine = torch.remainder(target_anchors, M4E_TICKS_PER_BEAT)
        gate_logit = (
            self.hidden_gate(self.hidden_normalization(hidden)).squeeze(-1)
            + self.beat_gate(beat).squeeze(-1)
            + self.fine_gate(fine).squeeze(-1)
            + self.gate_bias
        )
        gate = torch.sigmoid(gate_logit)
        note_row = batch.target_type == TARGET_NOTE_GROUP
        usable = (
            valid_candidate
            & note_row[:, None]
            & (chord_root[:, None] != NO_CHORD_ROOT)
        )
        bias = (gate[:, None] * energy).masked_fill(~usable, 0.0)
        return bias, {
            "gate": gate,
            "target_anchor_ticks": target_anchors,
            "chord_root": chord_root,
            "chord_quality": chord_quality,
            "usable_candidate_count": usable.sum(dim=-1),
        }


class FrozenMelodyHarmonicEnergyModel(nn.Module):
    """Frozen melody prior plus a trainable Anchor-logit energy adapter."""

    def __init__(
        self,
        *,
        base: HarmonicConditionedMelodyModel,
        chord_quality_classes: int,
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.energy = CandidateWiseHarmonicEnergy(
            width=self.base.width,
            chord_quality_classes=chord_quality_classes,
        )

    def adapter_parameters(self):
        return self.energy.parameters()

    def losses_with_chords(
        self, batch: MelodyCausalBatch, chord: ChordConditionBatch
    ) -> tuple[Mapping[str, torch.Tensor], dict[str, torch.Tensor]]:
        hidden = self.base.causal_hidden(batch)
        bias, receipt = self.energy(hidden, batch, chord)
        losses = self.base.decoder.losses(
            hidden, batch, anchor_logit_bias=bias
        )
        return losses, receipt

    def baseline_losses(
        self, batch: MelodyCausalBatch
    ) -> Mapping[str, torch.Tensor]:
        hidden = self.base.causal_hidden(batch)
        return self.base.decoder.losses(hidden, batch)
