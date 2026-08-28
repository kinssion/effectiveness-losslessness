from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import torch

from .iclr_formal_training import build_formal_model
from .iclr_matched_harness import RepresentationConfig
from .note_centric_music import INPUT_NOTE, NoteCausalBatch


TONAL_CANONICALIZATION_SCHEMA = "m4l.tonal_canonicalization.v1"
TONAL_ARMS = (
    "ARM_F_TRANSPOSED_ABSOLUTE",
    "ARM_G_TONAL_CANONICALIZED",
)
SHIFT_CANDIDATES = tuple(range(-5, 7))
SHIFT_NAMESPACE = "m4l.tonal_canonicalization.shift.v1"


def d_representation_config() -> RepresentationConfig:
    return RepresentationConfig(
        arm="ARM_D_RELATIONAL_TIME_GEOMETRY",
        content_pitch="raw_id",
        ordinary_sequence_position=False,
        absolute_time_lookup=False,
        unary_time_geometry=True,
        pairwise_time_bias=True,
        optional_reference=False,
        fixed_reference_pitch_class=None,
    )


def build_tonal_model(protocol: Mapping[str, Any], *, seed: int):
    return build_formal_model(d_representation_config(), protocol, seed=int(seed))


def legal_shifts(
    minimum_pitch: int,
    maximum_pitch: int,
    *,
    candidates: Sequence[int] = SHIFT_CANDIDATES,
) -> tuple[int, ...]:
    if not 0 <= int(minimum_pitch) <= int(maximum_pitch) <= 127:
        raise ValueError("source pitch bounds leave MIDI support")
    result = tuple(
        int(value)
        for value in candidates
        if 0 <= int(minimum_pitch) + int(value)
        and int(maximum_pitch) + int(value) <= 127
    )
    if not result:
        raise ValueError("no legal preregistered transposition")
    return result


def deterministic_shift(
    sample_id: str,
    *,
    domain: str,
    epoch: int,
    seed: int,
    occurrence_index: int,
    minimum_pitch: int,
    maximum_pitch: int,
    candidates: Sequence[int] = SHIFT_CANDIDATES,
) -> int:
    legal = legal_shifts(minimum_pitch, maximum_pitch, candidates=candidates)
    payload = (
        f"{SHIFT_NAMESPACE}|{domain}|{sample_id}|{int(epoch)}|{int(seed)}|"
        f"{int(occurrence_index)}"
    ).encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(legal)
    return int(legal[index])


def _batch_fields(batch: NoteCausalBatch) -> dict[str, torch.Tensor]:
    return {name: getattr(batch, name) for name in batch.__dataclass_fields__}


def _pitch_bounds(batch: NoteCausalBatch, row: int) -> tuple[int, int]:
    input_values = batch.input_pitch[row][batch.input_type[row] == INPUT_NOTE]
    target_values = batch.target_pitch[row][batch.note_mask[row]]
    values = torch.cat((input_values, target_values))
    if not values.numel():
        raise ValueError("sample has no sounding note")
    return int(values.min()), int(values.max())


def transform_pitch_coordinates(
    batch: NoteCausalBatch,
    shifts: torch.Tensor,
    *,
    direction: str,
) -> NoteCausalBatch:
    if shifts.shape != (batch.valid.shape[0],):
        raise ValueError("one signed shift is required per sample")
    if direction not in ("forward", "inverse"):
        raise ValueError(f"unsupported transform direction: {direction}")
    sign = 1 if direction == "forward" else -1
    fields = _batch_fields(batch)
    input_pitch = batch.input_pitch.clone()
    target_pitch = batch.target_pitch.clone()
    for row, shift in enumerate(shifts.tolist()):
        input_notes = batch.input_type[row] == INPUT_NOTE
        target_notes = batch.note_mask[row]
        input_pitch[row, input_notes] += sign * int(shift)
        target_pitch[row, target_notes] += sign * int(shift)
        if bool(((input_pitch[row, input_notes] < 0) | (input_pitch[row, input_notes] > 127)).any()):
            raise ValueError("transformed input pitch leaves MIDI support")
        if bool(((target_pitch[row, target_notes] < 0) | (target_pitch[row, target_notes] > 127)).any()):
            raise ValueError("transformed target pitch leaves MIDI support")
    fields["input_pitch"] = input_pitch
    fields["target_pitch"] = target_pitch
    return NoteCausalBatch(**fields)


@dataclass(frozen=True, slots=True)
class TonalBatchView:
    """Tokenizer receipt. Only model_batch may enter model.forward()."""

    model_batch: NoteCausalBatch
    shifted_physical_batch: NoteCausalBatch
    shifts: torch.Tensor
    arm: str


def build_tonal_batch(
    raw_batch: NoteCausalBatch,
    *,
    sample_ids: Sequence[str],
    arm: str,
    domain: str,
    epoch: int,
    seed: int,
    occurrence_index: int = 0,
    fixed_shifts: Mapping[str, int] | None = None,
) -> TonalBatchView:
    if arm not in TONAL_ARMS:
        raise ValueError(f"unsupported tonal-canonicalization arm: {arm}")
    if len(sample_ids) != raw_batch.valid.shape[0]:
        raise ValueError("sample IDs do not match batch rows")
    selected: list[int] = []
    for row, sample_id in enumerate(sample_ids):
        minimum, maximum = _pitch_bounds(raw_batch, row)
        if fixed_shifts is None:
            shift = deterministic_shift(
                sample_id,
                domain=domain,
                epoch=epoch,
                seed=seed,
                occurrence_index=occurrence_index,
                minimum_pitch=minimum,
                maximum_pitch=maximum,
            )
        else:
            shift = int(fixed_shifts[sample_id])
            if shift not in legal_shifts(minimum, maximum):
                raise ValueError(f"frozen shift is illegal for {sample_id}")
        selected.append(shift)
    shifts = torch.tensor(selected, dtype=torch.long, device=raw_batch.valid.device)
    shifted = transform_pitch_coordinates(raw_batch, shifts, direction="forward")
    if arm == "ARM_F_TRANSPOSED_ABSOLUTE":
        model_batch = shifted
    else:
        model_batch = transform_pitch_coordinates(shifted, shifts, direction="inverse")
    return TonalBatchView(
        model_batch=model_batch,
        shifted_physical_batch=shifted,
        shifts=shifts,
        arm=arm,
    )


def cyclic_absolute_class_indices(shifts: torch.Tensor, *, classes: int = 128) -> torch.Tensor:
    """Map canonical class indices to shifted absolute labels bijectively.

    Actual source/target notes are never wrapped. The modulo completion applies
    only to the unused edge of the finite probability vocabulary so that the
    evaluator transformation is a normalization-preserving permutation.
    """

    base = torch.arange(classes, device=shifts.device)
    return torch.remainder(base[None, :] + shifts[:, None], classes)


def inverse_tokenizer_pitch(canonical_pitch: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    return canonical_pitch + shifts[:, None]
