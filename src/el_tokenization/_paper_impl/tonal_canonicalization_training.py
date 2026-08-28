from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .iclr_formal_training import sha256_file
from .iclr_matched_harness import (
    _decoder_predictions,
    parameter_receipt,
    source_tree_receipt,
)
from .note_centric_music import NoteCausalBatch
from .popk_clean_manifest import PopKManifestEntry
from .tonal_canonicalization import (
    SHIFT_CANDIDATES,
    TONAL_ARMS,
    build_tonal_batch,
)


FORMAL_SCHEMA = "m4l.tonal_canonicalization.formal_run.v1"


def load_tonal_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "m4l.tonal_canonicalization.formal_protocol.v1":
        raise ValueError("unsupported tonal-canonicalization protocol")
    if tuple(protocol["arms"]) != TONAL_ARMS:
        raise ValueError("tonal arm order differs from frozen contract")
    if tuple(protocol["preregistered_seeds"]) != (20260814, 20260815, 20260816):
        raise ValueError("formal seeds differ from frozen contract")
    if tuple(protocol["transposition"]["candidate_shifts_semitones"]) != SHIFT_CANDIDATES:
        raise ValueError("shift candidates differ from implementation")
    expected = (
        int(protocol["data"]["train_target_events_per_equivalent_epoch"])
        * int(protocol["training"]["target_equivalent_epochs"])
    )
    if int(protocol["training"]["target_event_budget"]) != expected:
        raise ValueError("target-event budget is not an exact epoch multiple")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("formal training must seal test evaluation")
    return protocol


def tonal_source_paths(root: Path) -> list[Path]:
    paths = [
        root / "unified_structured_music/tonal_canonicalization.py",
        root / "unified_structured_music/tonal_canonicalization_training.py",
        root / "unified_structured_music/iclr_matched_harness.py",
        root / "unified_structured_music/iclr_formal_training.py",
        root / "unified_structured_music/iclr_cached_training.py",
        root / "unified_structured_music/note_centric_music.py",
        root / "unified_structured_music/popk_clean_manifest.py",
        root / "tools/prepare_tonal_canonicalization_schedules.py",
        root / "tools/run_tonal_canonicalization_preflight.py",
        root / "tools/run_tonal_canonicalization_formal_arm_cached.py",
        root / "tools/freeze_tonal_canonicalization_validation.py",
        root / "tools/run_tonal_canonicalization_clean_test.py",
        root / "cloud/run_tonal_canonicalization_67.sh",
        root / "cloud/wait_hi_then_run_tonal_canonicalization_67.sh",
        root / "configs/tonal_canonicalization_v1.json",
        root / "artifacts/tonal_canonicalization_v1/transposition_schedule_spec.json",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"tonal-canonicalization source tree incomplete: {missing}")
    return paths


def tonal_source_tree_receipt(root: Path) -> dict[str, Any]:
    return source_tree_receipt(tonal_source_paths(root), repository_root=root)


def arm_config_receipt(protocol: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], str]:
    if arm not in TONAL_ARMS:
        raise ValueError(arm)
    payload = {
        "schema_version": "m4l.tonal_canonicalization.arm.v1",
        "arm": arm,
        "representation": dict(protocol["arms"][arm]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, __import__("hashlib").sha256(encoded).hexdigest()


def load_shift_manifest(path: Path, *, expected_split: str) -> tuple[dict[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m4l.tonal_canonicalization.fixed_shift_manifest.v1":
        raise ValueError("unsupported fixed shift manifest")
    if payload.get("split") != expected_split:
        raise ValueError("fixed shift manifest split mismatch")
    mapping = {str(row["sample_id"]): int(row["shift"]) for row in payload["rows"]}
    if len(mapping) != len(payload["rows"]):
        raise ValueError("duplicate sample in shift manifest")
    return mapping, payload


def _pitch_logits(model, hidden: torch.Tensor, batch: NoteCausalBatch) -> torch.Tensor:
    decoder = model.decoder
    state = decoder.normalization(hidden)
    state = decoder._advance(state, decoder.type_embedding(batch.target_type), batch.valid)
    state = decoder._advance(
        state, decoder.time_bar_embedding(batch.target_time_bars), batch.support_mask
    )
    state = decoder._advance(
        state,
        decoder.time_remainder_embedding(batch.target_time_remainders),
        batch.support_mask,
    )
    return decoder.pitch_head(decoder.normalization(state))


def evaluate_tonal(
    model,
    *,
    arm: str,
    entries: Sequence[PopKManifestEntry],
    materializer,
    batch_indices: Sequence[Sequence[int]],
    device: torch.device,
    fixed_shifts: Mapping[str, int],
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "type", "time", "pitch", "duration")}
    events = notes = support = eos_predictions = invalid_predictions = 0
    invalid_reconstructed = 0
    mapped_pitch_nll = native_pitch_nll = 0.0
    song_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for indices in batch_indices:
            selected = [entries[index] for index in indices]
            raw = materializer.batch(selected)
            view = build_tonal_batch(
                raw,
                sample_ids=tuple(entry.sample_id for entry in selected),
                arm=arm,
                domain=split,
                epoch=-1,
                seed=0,
                fixed_shifts=fixed_shifts,
            )
            batch = view.model_batch.to(device)
            shifts = view.shifts.to(device)
            shifted_physical = view.shifted_physical_batch.to(device)
            losses, _representation, hidden = model.losses(batch)
            logits = _pitch_logits(model, hidden, batch)
            native = F.cross_entropy(
                logits.reshape(-1, 128), batch.target_pitch.reshape(-1), reduction="none"
            ).reshape_as(batch.target_pitch) * batch.note_mask
            if arm == "ARM_G_TONAL_CANONICALIZED":
                # REST/EOS positions carry a placeholder pitch ID.  The inverse
                # coordinate transform is defined only for NOTE targets; keep
                # every masked placeholder inside the 128-class vocabulary so
                # cross-entropy remains valid before the note mask is applied.
                absolute_target_as_canonical = torch.where(
                    batch.note_mask,
                    shifted_physical.target_pitch - shifts[:, None],
                    batch.target_pitch,
                )
                mapped = F.cross_entropy(
                    logits.reshape(-1, 128),
                    absolute_target_as_canonical.reshape(-1),
                    reduction="none",
                ).reshape_as(batch.target_pitch) * batch.note_mask
                reconstructed = logits.argmax(dim=-1) + shifts[:, None]
                invalid_reconstructed += int(
                    (((reconstructed < 0) | (reconstructed > 127)) & batch.note_mask).sum()
                )
            else:
                mapped = native
            native_pitch_nll += float(native.sum().cpu())
            mapped_pitch_nll += float(mapped.sum().cpu())
            predictions = _decoder_predictions(model, hidden, batch)
            for row, entry in enumerate(selected):
                row_events = int(batch.valid[row].sum())
                row_notes = int(batch.note_mask[row].sum())
                row_support = int(batch.support_mask[row].sum())
                row_values = {
                    name: float(losses[name][row].sum().detach().cpu()) for name in totals
                }
                for name, value in row_values.items():
                    totals[name] += value
                song_rows.append(
                    {
                        "sample_id": entry.sample_id,
                        "shift": int(view.shifts[row]),
                        "event_count": row_events,
                        "note_count": row_notes,
                        "total_bits_per_event": row_values["total"] / max(1, row_events) / math.log(2.0),
                        "pitch_nll_nats_per_note": row_values["pitch"] / max(1, row_notes),
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
    metrics = {
        "primary_metric_name": "micro_total_bits_per_exact_shifted_musical_event",
        "total_bits_per_event": totals["total"] / max(1, events) / math.log(2.0),
        "type_nll_nats_per_event": totals["type"] / max(1, events),
        "time_nll_nats_per_support_event": totals["time"] / max(1, support),
        "pitch_nll_nats_per_note": mapped_pitch_nll / max(1, notes),
        "native_pitch_nll_nats_per_note": native_pitch_nll / max(1, notes),
        "duration_nll_nats_per_support_event": totals["duration"] / max(1, support),
        "mapped_native_pitch_nll_abs_difference": abs(mapped_pitch_nll - native_pitch_nll),
        "predicted_eos_rate": eos_predictions / max(1, events),
        "invalid_event_rate": invalid_predictions / max(1, events),
        "invalid_reconstructed_pitch_rate": invalid_reconstructed / max(1, notes),
        "evaluated_event_count": events,
        "evaluated_note_count": notes,
        "evaluated_support_event_count": support,
        "evaluated_song_count": len(entries),
        "throughput_events_per_second": events / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    return metrics, song_rows


def run_id_for_tonal(arm: str, seed: int) -> str:
    return f"formal_{arm.removeprefix('ARM_').lower()}_seed{int(seed)}"


def required_tonal_receipt(
    *,
    run_id: str,
    arm: str,
    seed: int,
    protocol_sha256: str,
    config_sha256: str,
    manifest_sha256: str,
    source_tree_sha256: str,
    schedule_spec_sha256: str,
    validation_shift_manifest_sha256: str,
    model,
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
        "schedule_spec_sha256": schedule_spec_sha256,
        "validation_shift_manifest_sha256": validation_shift_manifest_sha256,
        "model_parameter_count": parameter_receipt(model),
        "representation_config": dict(representation_config),
        "target_event_budget": int(target_event_budget),
        "checkpoint_path": "PENDING_FORMAL_RUN",
        "hardware": "PENDING_FORMAL_RUN",
        "start_time_utc": "PENDING_FORMAL_RUN",
        "end_time_utc": "PENDING_FORMAL_RUN",
        "status": "LAUNCH_READY",
        "test_reads": 0,
    }
