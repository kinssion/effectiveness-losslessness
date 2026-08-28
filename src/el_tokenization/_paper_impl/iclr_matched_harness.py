from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import tracemalloc
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .discrete_music_primitive import (
    M4E_BEATS_PER_SEMANTIC_BAR,
    M4E_MAX_ABSOLUTE_BARS,
    M4E_TICKS_PER_BEAT,
    M4E_TICKS_PER_SEMANTIC_BAR,
)
from .dual_coordinate_melody import (
    RelativeCausalBlock,
    TIME_RELATION_BUCKETS,
    prepare_relative_causal_attention_mask,
    signed_log_bucket,
)
from .musical_time_melody import signed_unit_distance
from .note_centric_music import (
    INPUT_BOS,
    INPUT_NOTE,
    INPUT_REST,
    TARGET_EOS,
    NoteCausalBatch,
    NoteFactorizedDecoder,
    linearize_note_tokens,
    tensorize_note_streams,
)
from .popk_clean_manifest import PopKCleanManifest, PopKManifestEntry
from .popk_clean_split import _parse_midi_bytes
from .sparse_melody_bpe import MelodyAtom, SparseMelodyStream, SparseNote


HARNESS_SCHEMA = "m4l.iclr_matched_representation_harness.v1"
ARM_NAMES = (
    "ARM_A_RAW_SEQUENCE",
    "ARM_B_ABSOLUTE_MUSICAL_TIME",
    "ARM_C_UNARY_TIME_GEOMETRY",
    "ARM_D_RELATIONAL_TIME_GEOMETRY",
    "ARM_E_FULL_COORDINATE",
    "ARM_F_OPTIONAL_REFERENCE",
)


@dataclass(frozen=True, slots=True)
class RepresentationConfig:
    arm: str
    content_pitch: str
    ordinary_sequence_position: bool
    absolute_time_lookup: bool
    unary_time_geometry: bool
    pairwise_time_bias: bool
    optional_reference: bool
    fixed_reference_pitch_class: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RepresentationConfig":
        result = cls(
            arm=str(value["arm"]),
            content_pitch=str(value["content_pitch"]),
            ordinary_sequence_position=bool(value["ordinary_sequence_position"]),
            absolute_time_lookup=bool(value["absolute_time_lookup"]),
            unary_time_geometry=bool(value["unary_time_geometry"]),
            pairwise_time_bias=bool(value["pairwise_time_bias"]),
            optional_reference=bool(value["optional_reference"]),
            fixed_reference_pitch_class=(
                None
                if value.get("fixed_reference_pitch_class") is None
                else int(value["fixed_reference_pitch_class"])
            ),
        )
        result.assert_contract()
        return result

    def assert_contract(self) -> None:
        if self.arm not in ARM_NAMES:
            raise ValueError(f"unsupported matched arm: {self.arm}")
        if self.content_pitch not in {
            "raw_id",
            "cyclic_pc_register",
            "reference_relative_cyclic_pc_register",
        }:
            raise ValueError(f"unsupported pitch representation: {self.content_pitch}")
        if self.arm == "ARM_A_RAW_SEQUENCE":
            expected = (True, False, False, False, False, "raw_id")
        elif self.arm == "ARM_B_ABSOLUTE_MUSICAL_TIME":
            expected = (True, True, False, False, False, "raw_id")
        elif self.arm == "ARM_C_UNARY_TIME_GEOMETRY":
            expected = (False, False, True, False, False, "raw_id")
        elif self.arm == "ARM_D_RELATIONAL_TIME_GEOMETRY":
            expected = (False, False, True, True, False, "raw_id")
        elif self.arm == "ARM_E_FULL_COORDINATE":
            expected = (
                False,
                False,
                True,
                True,
                False,
                "cyclic_pc_register",
            )
        else:
            expected = (
                False,
                False,
                True,
                True,
                True,
                "reference_relative_cyclic_pc_register",
            )
        observed = (
            self.ordinary_sequence_position,
            self.absolute_time_lookup,
            self.unary_time_geometry,
            self.pairwise_time_bias,
            self.optional_reference,
            self.content_pitch,
        )
        if observed != expected:
            raise ValueError(
                f"information firewall mismatch for {self.arm}: {observed} != {expected}"
            )
        if self.optional_reference and self.fixed_reference_pitch_class is None:
            raise ValueError("optional reference arm requires an explicit reference policy")
        if self.fixed_reference_pitch_class is not None and not 0 <= self.fixed_reference_pitch_class < 12:
            raise ValueError("fixed reference pitch class leaves support")


@dataclass(frozen=True, slots=True)
class RepresentationOutput:
    content_features: torch.Tensor
    unary_time_features: torch.Tensor
    pairwise_attention_bias: torch.Tensor
    optional_reference_features: torch.Tensor
    metadata: Mapping[str, Any]


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load the repository's JSON-compatible YAML contracts without PyYAML."""

    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_sha256(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def target_tensor_sha256(batch: NoteCausalBatch) -> str:
    names = (
        "target_type",
        "target_time_bars",
        "target_time_remainders",
        "target_pitch",
        "target_duration_bars",
        "target_duration_remainders",
        "note_mask",
        "support_mask",
        "valid",
    )
    return tensor_sha256({name: getattr(batch, name) for name in names})


def module_state_sha256(module: nn.Module) -> str:
    return tensor_sha256(module.state_dict())


def source_tree_receipt(paths: Sequence[Path], *, repository_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for path in sorted({Path(path) for path in paths}, key=lambda item: item.as_posix()):
        payload = path.read_bytes()
        relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
        file_hash = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        rows.append({"path": relative, "sha256": file_hash, "bytes": len(payload)})
    return {
        "algorithm": "sha256(path NUL file_sha256 LF)",
        "source_tree_sha256": digest.hexdigest(),
        "files": rows,
        "git_revision": None,
    }


def _stream_from_notes(
    sample_id: str,
    notes: Sequence[Any],
    *,
    minimum_rest_ticks: int = M4E_TICKS_PER_BEAT,
) -> SparseMelodyStream:
    grouped: dict[int, list[SparseNote]] = defaultdict(list)
    for note in notes:
        grouped[int(note.onset_tick)].append(
            SparseNote.from_midi(int(note.pitch), int(note.duration_tick))
        )
    atoms: list[MelodyAtom] = []
    anchors: list[int] = []
    active_until: int | None = None
    for onset in sorted(grouped):
        group = sorted(grouped[onset], key=lambda note: (note.midi_pitch, note.duration_ticks))
        if active_until is not None and onset - active_until >= int(minimum_rest_ticks):
            atoms.append(MelodyAtom.rest(onset - active_until))
            anchors.append(active_until)
        atoms.append(MelodyAtom.note_group(group))
        anchors.append(onset)
        group_end = max(onset + note.duration_ticks for note in group)
        active_until = group_end if active_until is None else max(active_until, group_end)
    stream = SparseMelodyStream(
        song_id=sample_id,
        source_stream=0,
        atoms=tuple(atoms),
        anchors=tuple(anchors),
    )
    linearize_note_tokens(stream)
    return stream


class PopKManifestDataset:
    def __init__(self, manifest: PopKCleanManifest) -> None:
        self.manifest = manifest
        self.accepted = 0
        self.rejected = 0

    def stream(self, entry: PopKManifestEntry) -> SparseMelodyStream:
        payload = self.manifest.source_path(entry).read_bytes()
        notes, metadata = _parse_midi_bytes(payload)
        if int(metadata["ticks_per_beat"]) != M4E_TICKS_PER_BEAT:
            raise ValueError(
                f"{entry.sample_id} PPQ {metadata['ticks_per_beat']} != {M4E_TICKS_PER_BEAT}"
            )
        stream = _stream_from_notes(entry.sample_id, notes)
        self.accepted += 1
        return stream

    def materialize(self, entries: Sequence[PopKManifestEntry]) -> list[SparseMelodyStream]:
        streams: list[SparseMelodyStream] = []
        for entry in entries:
            try:
                streams.append(self.stream(entry))
            except Exception:
                self.rejected += 1
                raise
        return streams


class MatchedRepresentationEncoder(nn.Module):
    def __init__(
        self,
        config: RepresentationConfig,
        *,
        width: int,
        heads: int,
        maximum_sequence_tokens: int,
        maximum_context_bars: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.width = int(width)
        self.heads = int(heads)
        self.maximum_sequence_tokens = int(maximum_sequence_tokens)
        self.maximum_context_bars = int(maximum_context_bars)
        self.input_type = nn.Embedding(4, width)
        self.raw_pitch = nn.Embedding(128, width)
        self.pitch_circle = nn.Linear(2, width, bias=False)
        self.register = nn.Embedding(11, width)
        self.duration_bars = nn.Embedding(256, width)
        self.duration_remainders = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, width)
        self.content_projection = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.unary_time_projection = nn.Sequential(
            nn.Linear(13, width), nn.SiLU(), nn.LayerNorm(width)
        )
        absolute_buckets = maximum_context_bars * 16 + 1
        self.absolute_onset_lookup = nn.Embedding(absolute_buckets, 4)
        self.absolute_bar_lookup = nn.Embedding(M4E_MAX_ABSOLUTE_BARS, 2)
        self.absolute_time_projection = nn.Sequential(
            nn.Linear(6, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.optional_reference_projection = nn.Sequential(
            nn.Linear(2, width), nn.SiLU(), nn.LayerNorm(width)
        )
        self.time_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.meter_bias = nn.Embedding(M4E_TICKS_PER_SEMANTIC_BAR, heads)
        self.bar_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        self.beat_distance_bias = nn.Embedding(TIME_RELATION_BUCKETS, heads)
        for embedding in (
            self.time_bias,
            self.meter_bias,
            self.bar_distance_bias,
            self.beat_distance_bias,
        ):
            nn.init.zeros_(embedding.weight)
        self._set_information_firewall()

    def _set_information_firewall(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        always = (
            self.input_type,
            self.duration_bars,
            self.duration_remainders,
            self.content_projection,
        )
        for module in always:
            module.requires_grad_(True)
        if self.config.content_pitch == "raw_id":
            self.raw_pitch.requires_grad_(True)
        else:
            self.pitch_circle.requires_grad_(True)
            self.register.requires_grad_(True)
        if self.config.absolute_time_lookup:
            self.absolute_onset_lookup.requires_grad_(True)
            self.absolute_bar_lookup.requires_grad_(True)
            self.absolute_time_projection.requires_grad_(True)
        if self.config.unary_time_geometry:
            self.unary_time_projection.requires_grad_(True)
        if self.config.pairwise_time_bias:
            self.time_bias.requires_grad_(True)
            self.meter_bias.requires_grad_(True)
            self.bar_distance_bias.requires_grad_(True)
            self.beat_distance_bias.requires_grad_(True)
        if self.config.optional_reference:
            self.optional_reference_projection.requires_grad_(True)

    @staticmethod
    def _phase(value: torch.Tensor, period: int, dtype: torch.dtype) -> torch.Tensor:
        angle = value.to(dtype) * (2.0 * math.pi / float(period))
        return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    def _pitch_features(self, batch: NoteCausalBatch) -> torch.Tensor:
        sounding = batch.input_type == INPUT_NOTE
        if self.config.content_pitch == "raw_id":
            pitch = self.raw_pitch(batch.input_pitch)
        else:
            dtype = self.pitch_circle.weight.dtype
            reference = int(self.config.fixed_reference_pitch_class or 0)
            pitch_class = torch.remainder(batch.input_pitch - reference, 12)
            pitch = self.pitch_circle(self._phase(pitch_class, 12, dtype))
            pitch = pitch + self.register(
                torch.div(batch.input_pitch, 12, rounding_mode="floor").clamp(0, 10)
            )
        return torch.where(sounding.unsqueeze(-1), pitch, torch.zeros_like(pitch))

    def content(self, batch: NoteCausalBatch) -> torch.Tensor:
        duration = self.duration_bars(batch.input_duration_bars) + self.duration_remainders(
            batch.input_duration_remainders
        )
        has_support = (batch.input_type == INPUT_NOTE) | (batch.input_type == INPUT_REST)
        values = self.input_type(batch.input_type) + self._pitch_features(batch)
        values = values + torch.where(
            has_support.unsqueeze(-1), duration, torch.zeros_like(duration)
        )
        return self.content_projection(values).masked_fill(~batch.valid.unsqueeze(-1), 0.0)

    def _sequence_position(self, batch: NoteCausalBatch, dtype: torch.dtype) -> torch.Tensor:
        length = batch.valid.shape[1]
        position = torch.arange(length, device=batch.valid.device, dtype=dtype)
        half = max(1, self.width // 2)
        scale = torch.exp(
            torch.arange(half, device=batch.valid.device, dtype=dtype)
            * (-math.log(10_000.0) / max(1, half - 1))
        )
        angle = position[:, None] * scale[None, :]
        values = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)[:, : self.width]
        if values.shape[-1] < self.width:
            values = torch.nn.functional.pad(values, (0, self.width - values.shape[-1]))
        return values.unsqueeze(0).expand(batch.valid.shape[0], -1, -1).masked_fill(
            ~batch.valid.unsqueeze(-1), 0.0
        )

    def unary_time(self, batch: NoteCausalBatch) -> torch.Tensor:
        dtype = self.content_projection[1].weight.dtype
        zeros = torch.zeros(
            (*batch.valid.shape, self.width), device=batch.valid.device, dtype=dtype
        )
        result = zeros
        if self.config.ordinary_sequence_position:
            result = result + self._sequence_position(batch, dtype)
        if self.config.absolute_time_lookup:
            anchor = batch.input_anchor_ticks.clamp_min(0)
            sixteenth = torch.div(anchor, M4E_TICKS_PER_BEAT // 4, rounding_mode="floor")
            absolute = torch.cat(
                (
                    self.absolute_onset_lookup(
                        sixteenth.clamp(0, self.absolute_onset_lookup.num_embeddings - 1)
                    ),
                    self.absolute_bar_lookup(
                        torch.div(anchor, M4E_TICKS_PER_SEMANTIC_BAR, rounding_mode="floor").clamp(
                            0, M4E_MAX_ABSOLUTE_BARS - 1
                        )
                    ),
                ),
                dim=-1,
            )
            result = result + self.absolute_time_projection(absolute)
        if self.config.unary_time_geometry:
            anchor = batch.input_anchor_ticks.clamp_min(0)
            bar = torch.div(anchor, M4E_TICKS_PER_SEMANTIC_BAR, rounding_mode="floor").to(dtype)
            normalized = bar / float(max(1, M4E_MAX_ABSOLUTE_BARS - 1))
            ordered = torch.stack(
                (
                    normalized,
                    torch.log1p(bar) / math.log(float(M4E_MAX_ABSOLUTE_BARS)),
                    torch.sqrt(normalized.clamp_min(0.0)),
                ),
                dim=-1,
            )
            features = torch.cat(
                (
                    ordered,
                    self._phase(
                        torch.remainder(anchor, M4E_TICKS_PER_BEAT),
                        M4E_TICKS_PER_BEAT,
                        dtype,
                    ),
                    self._phase(
                        torch.div(anchor, M4E_TICKS_PER_BEAT, rounding_mode="floor"),
                        M4E_BEATS_PER_SEMANTIC_BAR,
                        dtype,
                    ),
                    self._phase(anchor, M4E_TICKS_PER_SEMANTIC_BAR, dtype),
                    self._phase(anchor, 4 * M4E_TICKS_PER_SEMANTIC_BAR, dtype),
                    self._phase(anchor, 16 * M4E_TICKS_PER_SEMANTIC_BAR, dtype),
                ),
                dim=-1,
            )
            result = result + self.unary_time_projection(features)
        return result.masked_fill(~batch.valid.unsqueeze(-1), 0.0)

    def pairwise_bias(self, batch: NoteCausalBatch) -> torch.Tensor:
        batch_size, length = batch.valid.shape
        if not self.config.pairwise_time_bias:
            return torch.zeros(
                (batch_size, self.heads, length, length),
                device=batch.valid.device,
                dtype=self.content_projection[1].weight.dtype,
            )
        delta = batch.input_anchor_ticks[:, :, None] - batch.input_anchor_ticks[:, None, :]
        onset = self.time_bias(signed_log_bucket(delta))
        meter = self.meter_bias(torch.remainder(delta, M4E_TICKS_PER_SEMANTIC_BAR))
        bar = self.bar_distance_bias(
            signed_log_bucket(
                signed_unit_distance(delta, M4E_TICKS_PER_SEMANTIC_BAR),
                exact_magnitudes=8,
                maximum_distance=M4E_MAX_ABSOLUTE_BARS,
            )
        )
        beat = self.beat_distance_bias(
            signed_log_bucket(
                signed_unit_distance(delta, M4E_TICKS_PER_BEAT),
                exact_magnitudes=16,
                maximum_distance=M4E_MAX_ABSOLUTE_BARS * M4E_BEATS_PER_SEMANTIC_BAR,
            )
        )
        return (onset + meter + bar + beat).permute(0, 3, 1, 2)

    def optional_reference(self, batch: NoteCausalBatch) -> torch.Tensor:
        dtype = self.content_projection[1].weight.dtype
        if not self.config.optional_reference:
            return torch.zeros(
                (*batch.valid.shape, self.width), device=batch.valid.device, dtype=dtype
            )
        reference = torch.full_like(
            batch.input_anchor_ticks, int(self.config.fixed_reference_pitch_class)
        )
        return self.optional_reference_projection(self._phase(reference, 12, dtype)).masked_fill(
            ~batch.valid.unsqueeze(-1), 0.0
        )

    def forward(self, batch: NoteCausalBatch) -> RepresentationOutput:
        metadata = {
            "arm": self.config.arm,
            "content_pitch": self.config.content_pitch,
            "consumed_fields": {
                "event_type": True,
                "raw_pitch_identity": self.config.content_pitch == "raw_id",
                "cyclic_pitch_class": self.config.content_pitch != "raw_id",
                "absolute_register": self.config.content_pitch != "raw_id",
                "raw_duration_identity": True,
                "generic_sequence_position": self.config.ordinary_sequence_position,
                "absolute_time_lookup": self.config.absolute_time_lookup,
                "unary_musical_time": self.config.unary_time_geometry,
                "pairwise_musical_time": self.config.pairwise_time_bias,
                "optional_reference": self.config.optional_reference,
            },
        }
        return RepresentationOutput(
            content_features=self.content(batch),
            unary_time_features=self.unary_time(batch),
            pairwise_attention_bias=self.pairwise_bias(batch),
            optional_reference_features=self.optional_reference(batch),
            metadata=metadata,
        )


class UnifiedMatchedMusicModel(nn.Module):
    def __init__(
        self,
        representation: RepresentationConfig,
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
        self.representation = MatchedRepresentationEncoder(
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
        self, batch: NoteCausalBatch
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
        self, batch: NoteCausalBatch
    ) -> tuple[dict[str, torch.Tensor], RepresentationOutput, torch.Tensor]:
        hidden, representation = self.hidden_sequence(batch)
        return self.decoder.losses(hidden, batch), representation, hidden


def parameter_receipt(model: UnifiedMatchedMusicModel) -> dict[str, int | float]:
    def count(module: nn.Module, *, active_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not active_only or parameter.requires_grad
        )

    total = count(model)
    effective = count(model, active_only=True)
    representation_total = count(model.representation)
    representation_active = count(model.representation, active_only=True)
    backbone = count(model.blocks) + count(model.input_fusion) + count(model.output_normalization)
    decoder = count(model.decoder)
    return {
        "total_parameters": total,
        "effective_trainable_parameters": effective,
        "representation_parameters_total": representation_total,
        "representation_parameters_active": representation_active,
        "backbone_parameters": backbone,
        "decoder_parameters": decoder,
        "inactive_parameters": total - effective,
    }


def enabled_module_names(config: RepresentationConfig) -> tuple[str, ...]:
    names = [
        "input_type",
        "duration_bars",
        "duration_remainders",
        "content_projection",
    ]
    if config.content_pitch == "raw_id":
        names.append("raw_pitch")
    else:
        names.extend(("pitch_circle", "register"))
    if config.absolute_time_lookup:
        names.extend(("absolute_onset_lookup", "absolute_bar_lookup", "absolute_time_projection"))
    if config.unary_time_geometry:
        names.append("unary_time_projection")
    if config.pairwise_time_bias:
        names.extend(("time_bias", "meter_bias", "bar_distance_bias", "beat_distance_bias"))
    if config.optional_reference:
        names.append("optional_reference_projection")
    return tuple(names)


def gradient_path_audit(
    model: UnifiedMatchedMusicModel,
) -> dict[str, Any]:
    enabled = set(enabled_module_names(model.representation_config))
    enabled_status: dict[str, bool] = {}
    disabled_status: dict[str, bool] = {}
    for name, module in model.representation.named_children():
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        has_gradient = bool(gradients) and any(
            gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0
            for gradient in gradients
        )
        if name in enabled:
            enabled_status[name] = has_gradient
        else:
            disabled_status[name] = all(
                parameter.grad is None for parameter in module.parameters()
            ) and all(not parameter.requires_grad for parameter in module.parameters())
    return {
        "enabled": enabled_status,
        "disabled": disabled_status,
        "all_enabled_reached": all(enabled_status.values()),
        "all_disabled_blocked": all(disabled_status.values()),
    }


def _decoder_predictions(
    model: UnifiedMatchedMusicModel,
    hidden: torch.Tensor,
    batch: NoteCausalBatch,
) -> dict[str, torch.Tensor]:
    decoder = model.decoder
    state = decoder.normalization(hidden)
    type_prediction = decoder.type_head(state).argmax(dim=-1)
    state = decoder._advance(state, decoder.type_embedding(batch.target_type), batch.valid)
    state = decoder._advance(
        state,
        decoder.time_bar_embedding(batch.target_time_bars),
        batch.support_mask,
    )
    state = decoder._advance(
        state,
        decoder.time_remainder_embedding(batch.target_time_remainders),
        batch.support_mask,
    )
    state = decoder._advance(
        state, decoder.pitch_embedding(batch.target_pitch), batch.note_mask
    )
    duration_bar = decoder.duration_bar_head(decoder.normalization(state)).argmax(dim=-1)
    state = decoder._advance(
        state,
        decoder.duration_bar_embedding(batch.target_duration_bars),
        batch.support_mask,
    )
    duration_remainder = decoder.duration_remainder_head(
        decoder.normalization(state)
    ).argmax(dim=-1)
    return {
        "type": type_prediction,
        "duration_bar": duration_bar,
        "duration_remainder": duration_remainder,
    }


def evaluate_model(
    model: UnifiedMatchedMusicModel,
    batches: Sequence[NoteCausalBatch],
    *,
    evaluated_song_count: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "type", "time", "pitch", "duration")}
    events = notes = support = eos_predictions = invalid_predictions = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()
    started = time.perf_counter()
    with torch.no_grad():
        for source_batch in batches:
            batch = source_batch.to(device)
            losses, _representation, hidden = model.losses(batch)
            for name in totals:
                totals[name] += float(losses[name].sum().detach().cpu())
            predictions = _decoder_predictions(model, hidden, batch)
            valid = batch.valid
            predicted_support = valid & (predictions["type"] != TARGET_EOS)
            invalid = predicted_support & (predictions["duration_bar"] == 0) & (
                predictions["duration_remainder"] == 0
            )
            events += int(valid.sum())
            notes += int(batch.note_mask.sum())
            support += int(batch.support_mask.sum())
            eos_predictions += int(((predictions["type"] == TARGET_EOS) & valid).sum())
            invalid_predictions += int(invalid.sum())
    elapsed = max(time.perf_counter() - started, 1e-9)
    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
        peak_source = "torch.cuda.max_memory_allocated"
    else:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_source = "python_tracemalloc_cpu_only"
    return {
        "total_bits_per_event": totals["total"] / max(1, events) / math.log(2.0),
        "type_nll_nats_per_event": totals["type"] / max(1, events),
        "time_nll_nats_per_support_event": totals["time"] / max(1, support),
        "pitch_nll_nats_per_note": totals["pitch"] / max(1, notes),
        "duration_nll_nats_per_support_event": totals["duration"] / max(1, support),
        "predicted_eos_rate": eos_predictions / max(1, events),
        "invalid_event_rate": invalid_predictions / max(1, events),
        "evaluated_event_count": events,
        "evaluated_note_count": notes,
        "evaluated_song_count": int(evaluated_song_count),
        "throughput_events_per_second": events / elapsed,
        "peak_memory_bytes": peak,
        "peak_memory_source": peak_source,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hardware_receipt(device: torch.device) -> dict[str, Any]:
    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_device_queried": False,
    }
    if device.type == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
        result["cuda_device_queried"] = True
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
