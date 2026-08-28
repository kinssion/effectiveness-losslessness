from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .dual_coordinate_melody import RelativeCausalBlock, prepare_relative_causal_attention_mask
from .iclr_matched_harness import (
    MatchedRepresentationEncoder,
    RepresentationConfig,
    RepresentationOutput,
    parameter_receipt,
    seed_everything,
)
from .note_centric_music import (
    INPUT_NOTE,
    INPUT_REST,
    NoteCausalBatch,
    NoteFactorizedDecoder,
)


FGHI_SCHEMA = "m4l.fghi.matched_pitch_interfaces.v1"
FGHI_ARMS = (
    "ARM_F_TRANSPOSED_ABSOLUTE",
    "ARM_G_TRANSPOSED_TONAL_CORRECTED",
    "ARM_H_FACTORIZED_PITCH_ID",
    "ARM_I_FIFTHS_CIRCLE_PITCH",
)
TRANSPOSITION_ARMS = FGHI_ARMS[:2]
PITCH_INTERFACE_ARMS = FGHI_ARMS[2:]
TRANSPOSITION_NAMESPACE = "m4l.fghi.transposition.v1"
DEFAULT_CANDIDATE_SHIFTS = tuple(range(-5, 7))
EVALUATION_EPOCH_SENTINEL = -1


@dataclass(frozen=True, slots=True)
class FGHIRepresentationConfig:
    arm: str
    family: str
    pitch_interface: str
    transposition: bool
    tonal_frame_input: bool
    unary_time_geometry: bool = True
    pairwise_time_bias: bool = True

    @classmethod
    def from_protocol(
        cls, protocol: Mapping[str, Any], arm: str
    ) -> "FGHIRepresentationConfig":
        if arm not in FGHI_ARMS:
            raise ValueError(f"unsupported F/G/H/I arm: {arm}")
        value = protocol["arms"][arm]
        result = cls(
            arm=arm,
            family=str(value["family"]),
            pitch_interface=str(value["pitch_interface"]),
            transposition=bool(value["transposition"]),
            tonal_frame_input=bool(value.get("tonal_frame_input", False)),
            unary_time_geometry=bool(value["unary_time_geometry"]),
            pairwise_time_bias=bool(value["pairwise_time_bias"]),
        )
        result.assert_contract()
        return result

    def assert_contract(self) -> None:
        expected = {
            "ARM_F_TRANSPOSED_ABSOLUTE": (
                "tonal_reference_v1",
                "raw_transposed_midi_id",
                True,
                False,
            ),
            "ARM_G_TRANSPOSED_TONAL_CORRECTED": (
                "tonal_reference_v1",
                "learned_relative_pc_plus_relative_register",
                True,
                True,
            ),
            "ARM_H_FACTORIZED_PITCH_ID": (
                "pitch_interface_v1",
                "learned_absolute_pc_plus_absolute_register",
                False,
                False,
            ),
            "ARM_I_FIFTHS_CIRCLE_PITCH": (
                "pitch_interface_v1",
                "fixed_fifths_circle_plus_absolute_register",
                False,
                False,
            ),
        }[self.arm]
        observed = (
            self.family,
            self.pitch_interface,
            self.transposition,
            self.tonal_frame_input,
        )
        if observed != expected:
            raise ValueError(f"F/G/H/I information firewall mismatch: {observed} != {expected}")
        if not self.unary_time_geometry or not self.pairwise_time_bias:
            raise ValueError("all F/G/H/I arms must preserve frozen D temporal geometry")


@dataclass(slots=True)
class FGHICausalBatch:
    input_type: torch.Tensor
    input_pitch: torch.Tensor
    input_duration_bars: torch.Tensor
    input_duration_remainders: torch.Tensor
    input_anchor_ticks: torch.Tensor
    valid: torch.Tensor
    target_type: torch.Tensor
    target_time_bars: torch.Tensor
    target_time_remainders: torch.Tensor
    target_pitch: torch.Tensor
    target_duration_bars: torch.Tensor
    target_duration_remainders: torch.Tensor
    note_mask: torch.Tensor
    support_mask: torch.Tensor
    tonal_frame_pc: torch.Tensor
    sample_shift: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.valid.sum().item())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "FGHICausalBatch":
        return FGHICausalBatch(
            **{
                name: getattr(self, name).to(device, non_blocking=non_blocking)
                for name in self.__dataclass_fields__
            }
        )

    def pin_memory(self) -> "FGHICausalBatch":
        return FGHICausalBatch(
            **{name: getattr(self, name).pin_memory() for name in self.__dataclass_fields__}
        )


def legal_candidate_shifts(
    minimum_pitch: int,
    maximum_pitch: int,
    *,
    candidates: Sequence[int] = DEFAULT_CANDIDATE_SHIFTS,
) -> tuple[int, ...]:
    if not 0 <= int(minimum_pitch) <= int(maximum_pitch) <= 127:
        raise ValueError("source pitch bounds leave MIDI support")
    legal = tuple(
        int(shift)
        for shift in candidates
        if 0 <= int(minimum_pitch) + int(shift)
        and int(maximum_pitch) + int(shift) <= 127
    )
    if not legal:
        raise ValueError("no preregistered shift is legal for this sample")
    return legal


def deterministic_transposition_shift(
    sample_id: str,
    epoch: int,
    seed: int,
    occurrence_index: int,
    *,
    minimum_pitch: int,
    maximum_pitch: int,
    candidates: Sequence[int] = DEFAULT_CANDIDATE_SHIFTS,
) -> int:
    """Choose a legal shift without taking arm identity or musical labels."""

    legal = legal_candidate_shifts(
        minimum_pitch, maximum_pitch, candidates=tuple(int(value) for value in candidates)
    )
    payload = (
        f"{TRANSPOSITION_NAMESPACE}|{sample_id}|{int(epoch)}|{int(seed)}|"
        f"{int(occurrence_index)}"
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(legal)
    return int(legal[index])


def _copy_batch_fields(batch: NoteCausalBatch) -> dict[str, torch.Tensor]:
    return {
        name: getattr(batch, name)
        for name in NoteCausalBatch.__dataclass_fields__
    }


def attach_fghi_context(
    batch: NoteCausalBatch,
    *,
    sample_ids: Sequence[str],
    epoch: int,
    seed: int,
    occurrence_index: int = 0,
    transpose: bool,
    base_reference_pc: int = 0,
    candidates: Sequence[int] = DEFAULT_CANDIDATE_SHIFTS,
) -> FGHICausalBatch:
    if len(sample_ids) != batch.valid.shape[0]:
        raise ValueError("sample IDs do not match batch rows")
    fields = _copy_batch_fields(batch)
    input_pitch = batch.input_pitch.clone()
    target_pitch = batch.target_pitch.clone()
    shifts: list[int] = []
    for row, sample_id in enumerate(sample_ids):
        pitches = target_pitch[row][batch.note_mask[row]]
        if pitches.numel() == 0:
            raise ValueError(f"sample has no sounding note: {sample_id}")
        shift = 0
        if transpose:
            shift = deterministic_transposition_shift(
                sample_id,
                epoch,
                seed,
                occurrence_index,
                minimum_pitch=int(pitches.min()),
                maximum_pitch=int(pitches.max()),
                candidates=candidates,
            )
        input_note = batch.input_type[row] == INPUT_NOTE
        target_note = batch.note_mask[row]
        input_pitch[row, input_note] += shift
        target_pitch[row, target_note] += shift
        if bool(((input_pitch[row, input_note] < 0) | (input_pitch[row, input_note] > 127)).any()):
            raise AssertionError("shifted input pitch leaves MIDI support")
        if bool(((target_pitch[row, target_note] < 0) | (target_pitch[row, target_note] > 127)).any()):
            raise AssertionError("shifted target pitch leaves MIDI support")
        shifts.append(int(shift))
    fields["input_pitch"] = input_pitch
    fields["target_pitch"] = target_pitch
    shift_tensor = torch.tensor(shifts, dtype=torch.long, device=batch.valid.device)
    frame = torch.remainder(shift_tensor + int(base_reference_pc), 12)
    frame = frame[:, None].expand_as(batch.input_pitch).clone()
    return FGHICausalBatch(
        **fields,
        tonal_frame_pc=frame,
        sample_shift=shift_tensor,
    )


def chromatic_circle_index(pitch_class: torch.Tensor) -> torch.Tensor:
    return torch.remainder(pitch_class, 12)


def fifths_circle_index(pitch_class: torch.Tensor) -> torch.Tensor:
    return torch.remainder(7 * pitch_class, 12)


class FGHIRepresentationEncoder(MatchedRepresentationEncoder):
    def __init__(
        self,
        config: FGHIRepresentationConfig,
        *,
        width: int,
        heads: int,
        maximum_sequence_tokens: int,
        maximum_context_bars: int,
    ) -> None:
        # D is used only as the construction envelope. It creates the exact same
        # modules, in the exact same order, as frozen D/E before the new firewall
        # is applied.
        surrogate = RepresentationConfig(
            arm="ARM_D_RELATIONAL_TIME_GEOMETRY",
            content_pitch="raw_id",
            ordinary_sequence_position=False,
            absolute_time_lookup=False,
            unary_time_geometry=True,
            pairwise_time_bias=True,
            optional_reference=False,
            fixed_reference_pitch_class=None,
        )
        super().__init__(
            surrogate,
            width=width,
            heads=heads,
            maximum_sequence_tokens=maximum_sequence_tokens,
            maximum_context_bars=maximum_context_bars,
        )
        self.fghi_config = config
        self._set_fghi_firewall()

    def _set_fghi_firewall(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.input_type,
            self.duration_bars,
            self.duration_remainders,
            self.content_projection,
            self.unary_time_projection,
            self.time_bias,
            self.meter_bias,
            self.bar_distance_bias,
            self.beat_distance_bias,
        ):
            module.requires_grad_(True)
        if self.fghi_config.arm in (
            "ARM_F_TRANSPOSED_ABSOLUTE",
            "ARM_G_TRANSPOSED_TONAL_CORRECTED",
            "ARM_H_FACTORIZED_PITCH_ID",
        ):
            self.raw_pitch.requires_grad_(True)
        if self.fghi_config.arm in (
            "ARM_G_TRANSPOSED_TONAL_CORRECTED",
            "ARM_H_FACTORIZED_PITCH_ID",
            "ARM_I_FIFTHS_CIRCLE_PITCH",
        ):
            self.register.requires_grad_(True)
        if self.fghi_config.arm == "ARM_I_FIFTHS_CIRCLE_PITCH":
            self.pitch_circle.requires_grad_(True)

    def _pitch_features(self, batch: FGHICausalBatch) -> torch.Tensor:
        sounding = batch.input_type == INPUT_NOTE
        arm = self.fghi_config.arm
        if arm == "ARM_F_TRANSPOSED_ABSOLUTE":
            pitch = self.raw_pitch(batch.input_pitch)
        elif arm == "ARM_G_TRANSPOSED_TONAL_CORRECTED":
            # Undo the full signed corpus transposition before factorizing pitch.
            # Both pitch class and register therefore live in the same fixed-C
            # coordinate frame and remain invariant under paired transposition.
            normalized_pitch = batch.input_pitch - batch.sample_shift[:, None]
            relative_pc = torch.remainder(normalized_pitch, 12)
            pitch = self.raw_pitch(relative_pc)
            pitch = pitch + self.register(
                torch.div(normalized_pitch, 12, rounding_mode="floor").clamp(0, 10)
            )
        elif arm == "ARM_H_FACTORIZED_PITCH_ID":
            pitch = self.raw_pitch(torch.remainder(batch.input_pitch, 12))
            pitch = pitch + self.register(
                torch.div(batch.input_pitch, 12, rounding_mode="floor").clamp(0, 10)
            )
        elif arm == "ARM_I_FIFTHS_CIRCLE_PITCH":
            dtype = self.pitch_circle.weight.dtype
            index = fifths_circle_index(batch.input_pitch)
            pitch = self.pitch_circle(self._phase(index, 12, dtype))
            pitch = pitch + self.register(
                torch.div(batch.input_pitch, 12, rounding_mode="floor").clamp(0, 10)
            )
        else:  # pragma: no cover - contract construction prevents this
            raise AssertionError(f"unhandled arm: {arm}")
        return torch.where(sounding.unsqueeze(-1), pitch, torch.zeros_like(pitch))

    def optional_reference(self, batch: FGHICausalBatch) -> torch.Tensor:
        dtype = self.content_projection[1].weight.dtype
        if self.fghi_config.arm != "ARM_G_TRANSPOSED_TONAL_CORRECTED":
            return torch.zeros(
                (*batch.valid.shape, self.width), device=batch.valid.device, dtype=dtype
            )
        # Rows 12..23 are a categorical bank for the full signed frame shift.
        # The frozen candidate set is exactly -5..+6, hence shift+5 is 0..11.
        # Unlike pitch-class modulo arithmetic, this preserves octave carry.
        shift_index = (batch.sample_shift + 5)[:, None].expand_as(batch.input_pitch)
        reference = self.raw_pitch(12 + shift_index)
        return reference.masked_fill(~batch.valid.unsqueeze(-1), 0.0)

    def forward(self, batch: FGHICausalBatch) -> RepresentationOutput:
        arm = self.fghi_config.arm
        return RepresentationOutput(
            content_features=self.content(batch),
            unary_time_features=self.unary_time(batch),
            pairwise_attention_bias=self.pairwise_bias(batch),
            optional_reference_features=self.optional_reference(batch),
            metadata={
                "schema_version": FGHI_SCHEMA,
                "arm": arm,
                "family": self.fghi_config.family,
                "pitch_interface": self.fghi_config.pitch_interface,
                "consumed_fields": {
                    "event_type": True,
                    "raw_transposed_pitch_id": arm == "ARM_F_TRANSPOSED_ABSOLUTE",
                    "categorical_absolute_pc": arm == "ARM_H_FACTORIZED_PITCH_ID",
                    "categorical_relative_pc": arm == "ARM_G_TRANSPOSED_TONAL_CORRECTED",
                    "categorical_unwrapped_frame_shift": arm
                    == "ARM_G_TRANSPOSED_TONAL_CORRECTED",
                    "relative_register": arm == "ARM_G_TRANSPOSED_TONAL_CORRECTED",
                    "absolute_register": arm
                    in ("ARM_H_FACTORIZED_PITCH_ID", "ARM_I_FIFTHS_CIRCLE_PITCH"),
                    "fifths_circle": arm == "ARM_I_FIFTHS_CIRCLE_PITCH",
                    "chromatic_circle": False,
                    "unary_musical_time": True,
                    "pairwise_musical_time": True,
                },
            },
        )


class FGHIUnifiedMatchedMusicModel(nn.Module):
    def __init__(
        self,
        representation: FGHIRepresentationConfig,
        *,
        width: int = 64,
        heads: int = 4,
        layers: int = 16,
        feedforward_width: int = 128,
        dropout: float = 0.0,
        maximum_sequence_tokens: int = 2048,
        maximum_context_bars: int = 32,
    ) -> None:
        super().__init__()
        self.representation_config = representation
        self.width = int(width)
        self.heads = int(heads)
        self.representation = FGHIRepresentationEncoder(
            representation,
            width=width,
            heads=heads,
            maximum_sequence_tokens=maximum_sequence_tokens,
            maximum_context_bars=maximum_context_bars,
        )
        self.input_fusion = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
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
        self.decoder = NoteFactorizedDecoder(width)

    def hidden_sequence(
        self, batch: FGHICausalBatch
    ) -> tuple[torch.Tensor, RepresentationOutput]:
        representation = self.representation(batch)
        values = self.input_fusion(
            torch.cat(
                (
                    representation.content_features,
                    representation.unary_time_features,
                    representation.optional_reference_features,
                ),
                dim=-1,
            )
        )
        attention_mask = prepare_relative_causal_attention_mask(
            relative_bias=representation.pairwise_attention_bias,
            valid=batch.valid,
            dtype=values.dtype,
        )
        for block in self.blocks:
            values = block(values, valid=batch.valid, attention_mask=attention_mask)
        return self.output_normalization(values), representation

    def losses(
        self, batch: FGHICausalBatch
    ) -> tuple[dict[str, torch.Tensor], RepresentationOutput, torch.Tensor]:
        hidden, representation = self.hidden_sequence(batch)
        return self.decoder.losses(hidden, batch), representation, hidden


def build_fghi_model(
    config: FGHIRepresentationConfig,
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> FGHIUnifiedMatchedMusicModel:
    seed_everything(seed)
    spec = protocol["model"]
    return FGHIUnifiedMatchedMusicModel(
        config,
        width=int(spec["width"]),
        heads=int(spec["heads"]),
        layers=int(spec["layers"]),
        feedforward_width=int(spec["feedforward_width"]),
        dropout=float(spec["dropout"]),
        maximum_sequence_tokens=int(spec["maximum_sequence_tokens"]),
        maximum_context_bars=int(protocol["context_policy"]["maximum_context_bars"]),
    )


def fghi_parameter_receipt(model: FGHIUnifiedMatchedMusicModel) -> dict[str, int | float]:
    return parameter_receipt(model)  # identical module envelope and counting rule


def pitch_interface_utilized_rows(config: FGHIRepresentationConfig) -> dict[str, Any]:
    if config.arm == "ARM_F_TRANSPOSED_ABSOLUTE":
        return {"raw_pitch_rows": 128, "register_rows": 0, "circle_projection": False}
    if config.arm == "ARM_G_TRANSPOSED_TONAL_CORRECTED":
        return {"raw_pitch_rows": 24, "register_rows": 11, "circle_projection": False}
    if config.arm == "ARM_H_FACTORIZED_PITCH_ID":
        return {"raw_pitch_rows": 12, "register_rows": 11, "circle_projection": False}
    return {"raw_pitch_rows": 0, "register_rows": 11, "circle_projection": True}
