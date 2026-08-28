from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .commu_ad_replication import ComMUManifestEntry
from .iclr_cached_training import SharedNoteTensorCache
from .iclr_formal_training import sha256_file, write_json_atomic
from .iclr_matched_harness import source_tree_receipt
from .note_centric_music import INPUT_NOTE, NoteCausalBatch
from .tonal_canonicalization import (
    SHIFT_CANDIDATES,
    TONAL_ARMS,
    deterministic_shift,
    legal_shifts,
)


SCHEMA_VERSION = "m4l.commu_tonal_canonicalization.formal_protocol.v1"


def load_commu_tonal_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported ComMU tonal-canonicalization protocol")
    if tuple(protocol["scientific_scope"]["arms"]) != TONAL_ARMS:
        raise ValueError("ComMU tonal arm order differs from frozen contract")
    if tuple(int(value) for value in protocol["preregistered_seeds"]) != (
        20260814,
        20260815,
        20260816,
    ):
        raise ValueError("ComMU tonal seeds differ from frozen contract")
    if tuple(protocol["transposition"]["candidate_shifts_semitones"]) != SHIFT_CANDIDATES:
        raise ValueError("ComMU shift candidates differ from implementation")
    per_epoch = int(protocol["data"]["train_target_events_per_equivalent_epoch"])
    epochs = int(protocol["training"]["target_equivalent_epochs"])
    if int(protocol["training"]["target_event_budget"]) != per_epoch * epochs:
        raise ValueError("ComMU tonal target budget is not an exact epoch multiple")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("ComMU tonal training must keep clean test sealed")
    if protocol["data"]["secondary_mode_policy"] != "A minor remains A minor and is not collapsed into C major":
        raise ValueError("ComMU secondary-mode policy changed")
    return protocol


def commu_tonal_source_paths(root: Path, protocol_path: Path) -> list[Path]:
    paths = [
        root / "unified_structured_music/commu_tonal_canonicalization.py",
        root / "unified_structured_music/commu_ad_replication.py",
        root / "unified_structured_music/tonal_canonicalization.py",
        root / "unified_structured_music/tonal_canonicalization_training.py",
        root / "unified_structured_music/iclr_matched_harness.py",
        root / "unified_structured_music/iclr_formal_training.py",
        root / "unified_structured_music/iclr_cached_training.py",
        root / "unified_structured_music/note_centric_music.py",
        root / "unified_structured_music/dual_coordinate_melody.py",
        root / "unified_structured_music/discrete_music_primitive.py",
        root / "unified_structured_music/musical_time_melody.py",
        root / "unified_structured_music/sparse_melody_bpe.py",
        protocol_path,
        root / "tools/run_commu_tonal_launch_gate.py",
        root / "tools/run_commu_tonal_formal_arm.py",
        root / "tools/freeze_commu_tonal_validation.py",
        root / "tools/run_commu_tonal_clean_test_once.py",
        root / "cloud/run_commu_tonal_canonicalization.sh",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"ComMU tonal source tree incomplete: {missing}")
    return paths


def commu_tonal_source_tree_receipt(root: Path, protocol_path: Path) -> dict[str, Any]:
    return source_tree_receipt(
        commu_tonal_source_paths(root, protocol_path), repository_root=root
    )


def arm_config_receipt(protocol: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], str]:
    if arm not in TONAL_ARMS:
        raise ValueError(arm)
    payload = {
        "schema_version": "m4l.commu_tonal_canonicalization.arm.v1",
        "arm": arm,
        "representation": dict(protocol["representation"][arm]),
        "shared_backbone_interface": protocol["representation"][
            "shared_backbone_interface"
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, hashlib.sha256(encoded).hexdigest()


def run_id_for_commu_tonal(arm: str, seed: int) -> str:
    return f"commu4x4_{arm.removeprefix('ARM_').lower()}_seed{int(seed)}"


def _pitch_bounds(batch: NoteCausalBatch, row: int) -> tuple[int, int]:
    values = torch.cat(
        (
            batch.input_pitch[row][batch.input_type[row] == INPUT_NOTE],
            batch.target_pitch[row][batch.note_mask[row]],
        )
    )
    if not values.numel():
        raise ValueError("ComMU sample has no sounding note")
    return int(values.min()), int(values.max())


def fixed_shift_rows(
    entries: Sequence[ComMUManifestEntry],
    cache: SharedNoteTensorCache,
    *,
    domain: str,
    epoch: int,
    seed: int,
    batch_size: int = 128,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(entries), int(batch_size)):
        selected = entries[start : start + int(batch_size)]
        batch = cache.batch(selected)
        for row, entry in enumerate(selected):
            minimum, maximum = _pitch_bounds(batch, row)
            legal = legal_shifts(minimum, maximum)
            shift = deterministic_shift(
                entry.sample_id,
                domain=domain,
                epoch=epoch,
                seed=seed,
                occurrence_index=0,
                minimum_pitch=minimum,
                maximum_pitch=maximum,
            )
            rows.append(
                {
                    "sample_id": entry.sample_id,
                    "minimum_pitch": minimum,
                    "maximum_pitch": maximum,
                    "legal_shifts": list(legal),
                    "shift": shift,
                }
            )
    return rows


def write_validation_shift_manifest(
    path: Path,
    *,
    protocol_path: Path,
    manifest_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    schedule_seed: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": "m4l.tonal_canonicalization.fixed_shift_manifest.v1",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": manifest_sha256,
        "split": "validation",
        "schedule_seed": int(schedule_seed),
        "arm_identity_consumed": False,
        "rows": [dict(row) for row in rows],
    }
    write_json_atomic(path, payload)
    return payload
