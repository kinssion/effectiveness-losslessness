from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .fghi_experiments import (
    DEFAULT_CANDIDATE_SHIFTS,
    FGHI_ARMS,
    FGHIRepresentationConfig,
    FGHICausalBatch,
    attach_fghi_context,
    build_fghi_model,
    fghi_parameter_receipt,
)
from .iclr_formal_training import (
    _decoder_predictions,
    load_json_yaml,
    sha256_file,
)
from .iclr_matched_harness import source_tree_receipt
from .popk_clean_manifest import PopKManifestEntry


FORMAL_SCHEMA = "m4l.fghi.formal_run.v1"


def load_fghi_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json_yaml(path)
    if protocol.get("schema_version") != "m4l.fghi.formal_protocol.v1":
        raise ValueError("unsupported F/G/H/I formal protocol schema")
    expected = (
        *protocol["families"]["tonal_reference_v1"]["new_arms"],
        *protocol["families"]["pitch_interface_v1"]["new_arms"],
    )
    if tuple(expected) != FGHI_ARMS:
        raise ValueError("formal F/G/H/I arm order differs from preregistration")
    if tuple(protocol["preregistered_seeds"]) != (20260814, 20260815, 20260816):
        raise ValueError("formal F/G/H/I seeds differ from preregistration")
    expected_budget = (
        int(protocol["data"]["train_target_events_per_equivalent_epoch"])
        * int(protocol["training"]["target_equivalent_epochs"])
    )
    if int(protocol["training"]["target_event_budget"]) != expected_budget:
        raise ValueError("F/G/H/I target-event budget is not an exact epoch multiple")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("F/G/H/I formal protocol does not seal test evaluation")
    if protocol["training"]["cuda_runtime"]["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise ValueError("F/G/H/I protocol lacks deterministic cuBLAS workspace")
    if tuple(protocol["transposition"]["candidate_shifts_semitones"]) != DEFAULT_CANDIDATE_SHIFTS:
        raise ValueError("transposition candidates differ from frozen implementation")
    if protocol["transposition"]["arm_identity_in_schedule"]:
        raise ValueError("transposition schedule illegally consumes arm identity")
    return protocol


def fghi_source_paths(root: Path) -> list[Path]:
    paths = [
        root / "unified_structured_music/fghi_experiments.py",
        root / "unified_structured_music/fghi_formal_training.py",
        root / "unified_structured_music/iclr_matched_harness.py",
        root / "unified_structured_music/iclr_formal_training.py",
        root / "unified_structured_music/iclr_cached_training.py",
        root / "unified_structured_music/note_centric_music.py",
        root / "unified_structured_music/dual_coordinate_melody.py",
        root / "unified_structured_music/discrete_music_primitive.py",
        root / "unified_structured_music/musical_time_melody.py",
        root / "unified_structured_music/popk_clean_manifest.py",
        root / "tools/run_fghi_formal_arm_cached.py",
        root / "tools/run_fghi_preflight.py",
        root / "tools/freeze_fghi_validation.py",
        root / "tools/run_fghi_clean_test.py",
        root / "cloud/run_fghi_formal_67.sh",
        root / "cloud/run_g_relative_register_v2_67.sh",
        root / "configs/fghi_formal_protocol_v1.json",
        root / "configs/fghi_formal_protocol_v2_relative_register.json",
        root / "artifacts/fghi/transposition_schedule_spec.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"F/G/H/I formal source tree is incomplete: {missing}")
    return paths


def fghi_source_tree_receipt(root: Path) -> dict[str, Any]:
    return source_tree_receipt(fghi_source_paths(root), repository_root=root)


def fghi_config_for_arm(
    protocol: Mapping[str, Any], arm: str
) -> tuple[FGHIRepresentationConfig, dict[str, Any], str]:
    config = FGHIRepresentationConfig.from_protocol(protocol, arm)
    payload = {
        "schema_version": "m4l.fghi.arm.v1",
        "representation": dict(protocol["arms"][arm]),
    }
    payload["representation"]["arm"] = arm
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return config, payload, __import__("hashlib").sha256(canonical).hexdigest()


def batch_for_arm(
    raw_batch,
    *,
    entries: Sequence[PopKManifestEntry],
    config: FGHIRepresentationConfig,
    epoch: int,
    seed: int,
    occurrence_index: int = 0,
    protocol: Mapping[str, Any],
) -> FGHICausalBatch:
    transposition = protocol["transposition"]
    return attach_fghi_context(
        raw_batch,
        sample_ids=tuple(entry.sample_id for entry in entries),
        epoch=epoch,
        seed=seed,
        occurrence_index=occurrence_index,
        transpose=config.transposition,
        base_reference_pc=int(transposition["base_tonic_policy"].startswith("fixed_C_axis") and 0),
        candidates=tuple(int(value) for value in transposition["candidate_shifts_semitones"]),
    )


def evaluate_fghi(
    model,
    *,
    config: FGHIRepresentationConfig,
    entries: Sequence[PopKManifestEntry],
    materializer,
    batch_indices: Sequence[Sequence[int]],
    device: torch.device,
    seed: int,
    protocol: Mapping[str, Any],
    evaluation_epoch: int = -1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "type", "time", "pitch", "duration")}
    events = notes = support = eos_predictions = invalid_predictions = 0
    shift_histogram = {str(value): 0 for value in DEFAULT_CANDIDATE_SHIFTS}
    shifted_notes = total_absolute_pitch_delta = 0
    minimum_shifted_pitch = 128
    maximum_shifted_pitch = -1
    song_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in batch_indices:
            selected = [entries[index] for index in indices]
            raw = materializer.batch(selected)
            batch = batch_for_arm(
                raw,
                entries=selected,
                config=config,
                epoch=evaluation_epoch,
                seed=seed,
                protocol=protocol,
            ).to(device)
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
                shift = int(batch.sample_shift[row])
                shift_histogram[str(shift)] += 1
                note_values = batch.target_pitch[row][batch.note_mask[row]]
                if note_values.numel():
                    minimum_shifted_pitch = min(minimum_shifted_pitch, int(note_values.min()))
                    maximum_shifted_pitch = max(maximum_shifted_pitch, int(note_values.max()))
                    shifted_notes += int(note_values.numel())
                    total_absolute_pitch_delta += abs(shift) * int(note_values.numel())
                song_rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "event_count": row_events,
                        "note_count": row_notes,
                        "support_event_count": row_support,
                        "sample_shift": shift,
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
            predicted_support = valid & (predictions["type"] != 2)
            invalid = predicted_support & (predictions["duration_bar"] == 0) & (
                predictions["duration_remainder"] == 0
            )
            eos_predictions += int(((predictions["type"] == 2) & valid).sum())
            invalid_predictions += int(invalid.sum())
    elapsed = max(time.perf_counter() - started, 1e-9)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    metrics = {
        "primary_metric_name": "micro_total_bits_per_exact_musical_event",
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
        "shift_coverage": {
            "histogram_by_song": shift_histogram,
            "shifted_note_count": shifted_notes,
            "mean_absolute_semitone_shift_per_note": total_absolute_pitch_delta / max(1, shifted_notes),
            "minimum_shifted_pitch": None if minimum_shifted_pitch == 128 else minimum_shifted_pitch,
            "maximum_shifted_pitch": None if maximum_shifted_pitch == -1 else maximum_shifted_pitch,
        },
    }
    return metrics, song_rows


def run_id_for_fghi(arm: str, seed: int) -> str:
    short = arm.removeprefix("ARM_").lower()
    return f"formal_{short}_seed{int(seed)}"


def required_fghi_receipt(
    *,
    run_id: str,
    arm: str,
    seed: int,
    family: str,
    protocol_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    source_tree_sha256: str,
    schedule_spec_sha256: str,
    model,
    representation_config: Mapping[str, Any],
    target_event_budget: int,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_SCHEMA + ".receipt",
        "run_id": run_id,
        "arm": arm,
        "family": family,
        "seed": int(seed),
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "manifest_sha256": manifest_sha256,
        "source_tree_sha256": source_tree_sha256,
        "schedule_spec_sha256": schedule_spec_sha256,
        "model_parameter_count": fghi_parameter_receipt(model),
        "representation_config": dict(representation_config),
        "target_event_budget": int(target_event_budget),
        "checkpoint_path": "PENDING_FORMAL_RUN",
        "hardware": "PENDING_FORMAL_RUN",
        "start_time_utc": "PENDING_FORMAL_RUN",
        "end_time_utc": "PENDING_FORMAL_RUN",
        "status": "LAUNCH_READY",
        "test_reads": 0,
    }
