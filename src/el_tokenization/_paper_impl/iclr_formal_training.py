from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .iclr_matched_harness import (
    ARM_NAMES,
    PopKManifestDataset,
    RepresentationConfig,
    UnifiedMatchedMusicModel,
    _decoder_predictions,
    canonical_json_sha256,
    gradient_path_audit,
    hardware_receipt,
    load_json_yaml,
    module_state_sha256,
    parameter_receipt,
    seed_everything,
    source_tree_receipt,
    target_tensor_sha256,
    tensorize_note_streams,
    utc_now,
)
from .popk_clean_manifest import (
    FROZEN_POPK_CLEAN_V1_SHA256,
    PopKCleanManifest,
    PopKManifestEntry,
)
from .popk_clean_split import _parse_midi_bytes
from .iclr_matched_harness import _stream_from_notes
from .note_centric_music import NoteCausalBatch, TARGET_EOS


FORMAL_SCHEMA = "m4l.iclr_matched_representation.formal_run.v1"
PRIMARY_ARMS = ARM_NAMES[:5]
ARM_FILE_NAMES = {
    "ARM_A_RAW_SEQUENCE": "arm_a_raw_sequence.yaml",
    "ARM_B_ABSOLUTE_MUSICAL_TIME": "arm_b_absolute_time.yaml",
    "ARM_C_UNARY_TIME_GEOMETRY": "arm_c_unary_time.yaml",
    "ARM_D_RELATIONAL_TIME_GEOMETRY": "arm_d_relational_time.yaml",
    "ARM_E_FULL_COORDINATE": "arm_e_full_coordinate.yaml",
    "ARM_F_OPTIONAL_REFERENCE": "arm_f_optional_reference.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json_yaml(path)
    if protocol.get("schema_version") != "m4l.iclr_matched_representation.formal_protocol.v1":
        raise ValueError("unsupported formal protocol schema")
    if tuple(protocol["scientific_scope"]["primary_arms"]) != PRIMARY_ARMS:
        raise ValueError("primary arm order differs from frozen A-E order")
    if tuple(protocol["preregistered_seeds"]) != (20260814, 20260815, 20260816):
        raise ValueError("formal seeds differ from preregistration")
    expected_budget = (
        int(protocol["data"]["train_target_events_per_equivalent_epoch"])
        * int(protocol["training"]["target_equivalent_epochs"])
    )
    if int(protocol["training"]["target_event_budget"]) != expected_budget:
        raise ValueError("formal target-event budget is not an exact epoch multiple")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("formal protocol does not seal test evaluation")
    cuda_runtime = protocol["training"].get("cuda_runtime", {})
    if cuda_runtime.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("formal protocol lacks the frozen deterministic cuBLAS workspace")
    return protocol


def formal_source_paths(root: Path, protocol: Mapping[str, Any]) -> list[Path]:
    paths = [
        root / "unified_structured_music/iclr_matched_harness.py",
        root / "unified_structured_music/iclr_formal_training.py",
        root / "unified_structured_music/popk_clean_manifest.py",
        root / "unified_structured_music/popk_clean_split.py",
        root / "unified_structured_music/note_centric_music.py",
        root / "unified_structured_music/dual_coordinate_melody.py",
        root / "unified_structured_music/discrete_music_primitive.py",
        root / "unified_structured_music/musical_time_melody.py",
        root / "unified_structured_music/sparse_melody_bpe.py",
        root / "tools/run_iclr_formal_launch_gate.py",
        root / "tools/run_iclr_formal_arm.py",
        root / "cloud/run_iclr_formal_ablation_67.sh",
        root / "configs/iclr_ablation/formal_protocol_v1.yaml",
        root / str(protocol["representation"]["shared_config"]),
    ]
    paths.extend(
        root / str(value)
        for value in protocol["representation"]["arm_configs"].values()
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal source tree is incomplete: {missing}")
    return paths


def representation_for_arm(
    root: Path, protocol: Mapping[str, Any], arm: str
) -> tuple[RepresentationConfig, dict[str, Any], str]:
    if arm not in protocol["representation"]["arm_configs"]:
        raise ValueError(f"arm is absent from formal protocol: {arm}")
    config_path = root / str(protocol["representation"]["arm_configs"][arm])
    arm_config = load_json_yaml(config_path)
    representation = RepresentationConfig.from_mapping(arm_config["representation"])
    if representation.arm != arm:
        raise ValueError(f"arm config mismatch: {representation.arm} != {arm}")
    return representation, arm_config, sha256_file(config_path)


def build_formal_model(
    representation: RepresentationConfig,
    protocol: Mapping[str, Any],
    *,
    seed: int,
) -> UnifiedMatchedMusicModel:
    seed_everything(seed)
    spec = protocol["model"]
    return UnifiedMatchedMusicModel(
        representation,
        width=int(spec["width"]),
        heads=int(spec["heads"]),
        layers=int(spec["layers"]),
        feedforward_width=int(spec["feedforward_width"]),
        dropout=float(spec["dropout"]),
        maximum_sequence_tokens=int(spec["maximum_sequence_tokens"]),
        maximum_context_bars=int(protocol["context_policy"]["maximum_context_bars"]),
    )


def derive_epoch_seed(protocol_sha256: str, seed: int, epoch_index: int) -> int:
    payload = f"{protocol_sha256}|{int(seed)}|{int(epoch_index)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def epoch_batches(
    entries: Sequence[PopKManifestEntry],
    *,
    protocol_sha256: str,
    seed: int,
    epoch_index: int,
    batch_songs: int,
    bucket_window: int,
) -> list[tuple[int, ...]]:
    rng = np.random.Generator(np.random.PCG64(derive_epoch_seed(protocol_sha256, seed, epoch_index)))
    permutation = rng.permutation(len(entries)).tolist()
    batches: list[tuple[int, ...]] = []
    for start in range(0, len(permutation), int(bucket_window)):
        window = permutation[start : start + int(bucket_window)]
        window.sort(key=lambda index: (int(entries[index].note_count) + 1, entries[index].sample_id))
        batches.extend(
            tuple(window[offset : offset + int(batch_songs)])
            for offset in range(0, len(window), int(batch_songs))
        )
    order = rng.permutation(len(batches)).tolist()
    return [batches[index] for index in order]


def validation_batches(
    entries: Sequence[PopKManifestEntry], *, batch_songs: int
) -> list[tuple[int, ...]]:
    ordered = sorted(
        range(len(entries)),
        key=lambda index: (int(entries[index].note_count) + 1, entries[index].sample_id),
    )
    return [
        tuple(ordered[start : start + int(batch_songs)])
        for start in range(0, len(ordered), int(batch_songs))
    ]


@dataclass(frozen=True, slots=True)
class SourceTask:
    sample_id: str
    path: str
    expected_notes: int
    expected_onsets: int
    maximum_sequence_tokens: int
    maximum_context_ticks: int


def _materialize_source_task(task: SourceTask):
    notes, metadata = _parse_midi_bytes(Path(task.path).read_bytes())
    if int(metadata["ticks_per_beat"]) != 96:
        raise ValueError(f"{task.sample_id} has non-frozen PPQ")
    if len(notes) != int(task.expected_notes):
        raise ValueError(f"{task.sample_id} note count differs from clean manifest")
    if len({note.onset_tick for note in notes}) != int(task.expected_onsets):
        raise ValueError(f"{task.sample_id} onset count differs from clean manifest")
    stream = _stream_from_notes(task.sample_id, notes)
    from .note_centric_music import linearize_note_tokens

    tokens = linearize_note_tokens(stream)
    if len(tokens) + 1 > int(task.maximum_sequence_tokens):
        raise ValueError(f"{task.sample_id} exceeds frozen sequence support")
    if tokens and max(int(token.anchor) for token in tokens) >= int(task.maximum_context_ticks):
        raise ValueError(f"{task.sample_id} exceeds frozen 32-bar context")
    return stream


class ParallelManifestMaterializer:
    def __init__(
        self,
        manifest: PopKCleanManifest,
        *,
        workers: int,
        maximum_sequence_tokens: int,
        maximum_context_bars: int,
    ) -> None:
        self.manifest = manifest
        self.maximum_sequence_tokens = int(maximum_sequence_tokens)
        self.maximum_context_ticks = int(maximum_context_bars) * 4 * 96
        self.pool = ProcessPoolExecutor(max_workers=int(workers))

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def materialize(self, entries: Sequence[PopKManifestEntry]) -> list[Any]:
        tasks = [
            SourceTask(
                sample_id=entry.sample_id,
                path=str(self.manifest.source_path(entry)),
                expected_notes=int(entry.note_count),
                expected_onsets=int(entry.onset_count),
                maximum_sequence_tokens=self.maximum_sequence_tokens,
                maximum_context_ticks=self.maximum_context_ticks,
            )
            for entry in entries
        ]
        return list(self.pool.map(_materialize_source_task, tasks, chunksize=8))

    def batch(self, entries: Sequence[PopKManifestEntry]) -> NoteCausalBatch:
        return tensorize_note_streams(self.materialize(entries))

    def __enter__(self) -> "ParallelManifestMaterializer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def learning_rate_for_projected_exposure(
    protocol: Mapping[str, Any], projected_exposure: int
) -> float:
    training = protocol["training"]
    optimizer = training["optimizer"]
    schedule = training["schedule"]
    base = float(optimizer["learning_rate"])
    minimum = float(optimizer["minimum_learning_rate"])
    warmup = int(schedule["warmup_target_events"])
    end = int(schedule["decay_end_target_events"])
    exposure = min(max(1, int(projected_exposure)), end)
    if exposure <= warmup:
        return base * exposure / max(1, warmup)
    progress = (exposure - warmup) / max(1, end - warmup)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _metric_totals() -> dict[str, float]:
    return {name: 0.0 for name in ("total", "type", "time", "pitch", "duration")}


def evaluate_formal(
    model: UnifiedMatchedMusicModel,
    *,
    entries: Sequence[PopKManifestEntry],
    materializer: ParallelManifestMaterializer,
    batch_indices: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals = _metric_totals()
    events = notes = support = eos_predictions = invalid_predictions = 0
    song_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in batch_indices:
            selected = [entries[index] for index in indices]
            batch = materializer.batch(selected).to(device)
            losses, _representation, hidden = model.losses(batch)
            predictions = _decoder_predictions(model, hidden, batch)
            for row, entry in enumerate(selected):
                row_events = int(batch.valid[row].sum())
                row_notes = int(batch.note_mask[row].sum())
                row_support = int(batch.support_mask[row].sum())
                row_values = {
                    name: float(losses[name][row].sum().detach().cpu())
                    for name in totals
                }
                for name, value in row_values.items():
                    totals[name] += value
                song_rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "event_count": row_events,
                        "note_count": row_notes,
                        "support_event_count": row_support,
                        "total_bits_per_event": row_values["total"] / max(1, row_events) / math.log(2.0),
                        "type_nll_nats_per_event": row_values["type"] / max(1, row_events),
                        "time_nll_nats_per_support_event": row_values["time"] / max(1, row_support),
                        "pitch_nll_nats_per_note": row_values["pitch"] / max(1, row_notes),
                        "duration_nll_nats_per_support_event": row_values["duration"] / max(1, row_support),
                    }
                )
                events += row_events
                notes += row_notes
                support += row_support
            valid = batch.valid
            predicted_support = valid & (predictions["type"] != TARGET_EOS)
            invalid = predicted_support & (predictions["duration_bar"] == 0) & (
                predictions["duration_remainder"] == 0
            )
            eos_predictions += int(((predictions["type"] == TARGET_EOS) & valid).sum())
            invalid_predictions += int(invalid.sum())
    elapsed = max(time.perf_counter() - started, 1e-9)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metrics = {
        "primary_metric_name": "micro_total_validation_bits_per_exact_musical_event",
        "total_bits_per_event": totals["total"] / max(1, events) / math.log(2.0),
        "type_nll_nats_per_event": totals["type"] / max(1, events),
        "time_nll_nats_per_support_event": totals["time"] / max(1, support),
        "pitch_nll_nats_per_note": totals["pitch"] / max(1, notes),
        "duration_nll_nats_per_support_event": totals["duration"] / max(1, support),
        "predicted_eos_rate": eos_predictions / max(1, events),
        "invalid_event_rate": invalid_predictions / max(1, events),
        "evaluated_event_count": events,
        "evaluated_note_count": notes,
        "evaluated_song_count": len(entries),
        "throughput_events_per_second": events / elapsed,
        "peak_memory_bytes": peak,
    }
    return metrics, song_rows


def optimizer_for_model(
    model: UnifiedMatchedMusicModel, protocol: Mapping[str, Any]
) -> torch.optim.Optimizer:
    spec = protocol["training"]["optimizer"]
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(spec["learning_rate"]),
        betas=tuple(float(value) for value in spec["betas"]),
        eps=float(spec["epsilon"]),
        weight_decay=float(spec["weight_decay"]),
    )


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def rng_state_receipt() -> dict[str, Any]:
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    # A checkpoint loaded with map_location=cuda also moves the saved CPU RNG
    # tensor. PyTorch's CPU generator requires an actual CPU ByteTensor.
    cpu_state = payload["torch_cpu_rng"].detach().to(
        device="cpu", dtype=torch.uint8
    )
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and payload.get("torch_cuda_rng"):
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in payload["torch_cuda_rng"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def run_id_for(arm: str, seed: int) -> str:
    short = arm.removeprefix("ARM_").lower()
    return f"formal_{short}_seed{int(seed)}"


def required_receipt_template(
    *,
    run_id: str,
    arm: str,
    seed: int,
    protocol_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    source_tree_sha256: str,
    model_parameter_count: Mapping[str, Any],
    representation_config: Mapping[str, Any],
    target_event_budget: int,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_SCHEMA + ".receipt",
        "run_id": run_id,
        "arm": arm,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "model_parameter_count": dict(model_parameter_count),
        "representation_config": dict(representation_config),
        "target_event_budget": int(target_event_budget),
        "checkpoint_path": "PENDING_FORMAL_RUN",
        "hardware": "PENDING_FORMAL_RUN",
        "start_time_utc": "PENDING_FORMAL_RUN",
        "end_time_utc": "PENDING_FORMAL_RUN",
        "status": "LAUNCH_READY",
        "test_reads": 0,
    }
