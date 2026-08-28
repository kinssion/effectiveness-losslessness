from __future__ import annotations

from dataclasses import fields
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_pop1k7_ad_formal_arm import (
    append_jsonl,
    batch_stream_sha256,
    checkpoint_payload,
    validation_milestones,
)
from .fghi_experiments import (
    FGHICausalBatch,
    FGHIRepresentationConfig,
    FGHIUnifiedMatchedMusicModel,
)
from .iclr_cached_training import SharedNoteTensorCache
from .iclr_formal_training import (
    epoch_batches,
    learning_rate_for_projected_exposure,
    optimizer_for_model,
    restore_rng_state,
    rng_state_receipt,
    save_checkpoint_atomic,
    sha256_file,
    validation_batches,
    write_json_atomic,
    write_jsonl_atomic,
)
from .iclr_matched_harness import (
    RepresentationConfig,
    hardware_receipt,
    parameter_receipt,
    seed_everything,
    source_tree_receipt,
    utc_now,
)
from .note_centric_music import INPUT_NOTE, NoteCausalBatch
from .pop1k7_ad_replication import Pop1K7CleanManifest, load_pop1k7_protocol
from .pop1k7_model_v2 import FastCausalUnifiedMatchedMusicModel
from .pop1k7_trainer_v2 import (
    FastCachedBatchPrefetcher,
    StaticValidationBatch,
    build_static_validation_batches,
    evaluate_cached_v2,
)
from .tonal_canonicalization import SHIFT_CANDIDATES, deterministic_shift


REPRESENTATION_ARMS = (
    "ARM_E_FULL_COORDINATE",
    "ARM_H_FACTORIZED_PITCH_ID",
    "ARM_I_FIFTHS_CIRCLE_PITCH",
)
TONAL_ARMS = (
    "ARM_F_TRANSPOSED_ABSOLUTE",
    "ARM_G_TONAL_CANONICALIZED",
)
SUPPORTED_ARMS = REPRESENTATION_ARMS + TONAL_ARMS


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_remaining_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "m4l.pop1k7_remaining.protocol.v1":
        raise ValueError("unsupported Pop1K7 remaining protocol")
    base_path = ROOT / str(protocol["base_protocol_path"])
    if sha256_file(base_path) != str(protocol["base_protocol_sha256"]):
        raise ValueError("frozen Pop1K7 base protocol changed")
    base = load_pop1k7_protocol(base_path)
    if int(protocol["training"]["target_event_budget"]) != int(
        base["training"]["target_event_budget"]
    ):
        raise ValueError("remaining and base target budgets differ")
    if int(protocol["training"]["target_equivalent_epochs"]) != int(
        base["training"]["target_equivalent_epochs"]
    ):
        raise ValueError("remaining and base epoch budgets differ")
    if protocol["validation_and_checkpointing"]["test_evaluation"] != "disabled_and_not_loaded":
        raise ValueError("Pop1K7 remaining protocol does not seal test")
    return protocol, base, base_path


def remaining_source_paths(protocol_path: Path, base_path: Path) -> list[Path]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    arms = set(protocol.get("arms", {}))
    paths = [
        protocol_path,
        base_path,
        ROOT / "unified_structured_music/pop1k7_remaining_training.py",
        ROOT / "unified_structured_music/pop1k7_ad_replication.py",
        ROOT / "unified_structured_music/pop1k7_model_v2.py",
        ROOT / "unified_structured_music/pop1k7_trainer_v2.py",
        ROOT / "unified_structured_music/fghi_experiments.py",
        ROOT / "unified_structured_music/tonal_canonicalization.py",
        ROOT / "unified_structured_music/iclr_matched_harness.py",
        ROOT / "unified_structured_music/iclr_formal_training.py",
        ROOT / "unified_structured_music/iclr_cached_training.py",
        ROOT / "unified_structured_music/note_centric_music.py",
        ROOT / "tools/run_pop1k7_ad_formal_arm.py",
        ROOT / "tools/run_pop1k7_remaining_representation_arm.py",
    ]
    if arms & set(TONAL_ARMS):
        paths.extend(
            [
                ROOT / "unified_structured_music/tonal_canonicalization.py",
                ROOT / "tools/run_pop1k7_remaining_tonal_arm.py",
            ]
        )
    if arms & {"J_LOSSLESS_REMI_LIKE", "K_TRAIN_ONLY_LOSSLESS_BPE"}:
        paths.extend(
            [
                ROOT / "unified_structured_music/external_tokenizers.py",
                ROOT / "unified_structured_music/external_tokenizer_training.py",
                ROOT / "unified_structured_music/pop1k7_external_model_v2.py",
                ROOT / "tools/run_pop1k7_remaining_external_arm.py",
            ]
        )
    launch_script = protocol.get("launch", {}).get("launch_script_path")
    if launch_script:
        paths.append(ROOT / str(launch_script))
    return paths


def _d_config() -> RepresentationConfig:
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


def _e_config() -> RepresentationConfig:
    return RepresentationConfig(
        arm="ARM_E_FULL_COORDINATE",
        content_pitch="cyclic_pc_register",
        ordinary_sequence_position=False,
        absolute_time_lookup=False,
        unary_time_geometry=True,
        pairwise_time_bias=True,
        optional_reference=False,
        fixed_reference_pitch_class=None,
    )


def _hi_config(arm: str) -> FGHIRepresentationConfig:
    if arm == "ARM_H_FACTORIZED_PITCH_ID":
        return FGHIRepresentationConfig(
            arm=arm,
            family="pitch_interface_v1",
            pitch_interface="learned_absolute_pc_plus_absolute_register",
            transposition=False,
            tonal_frame_input=False,
        )
    if arm == "ARM_I_FIFTHS_CIRCLE_PITCH":
        return FGHIRepresentationConfig(
            arm=arm,
            family="pitch_interface_v1",
            pitch_interface="fixed_fifths_circle_plus_absolute_register",
            transposition=False,
            tonal_frame_input=False,
        )
    raise ValueError(arm)


def build_model(arm: str, base: Mapping[str, Any], *, seed: int):
    seed_everything(int(seed))
    spec = base["model"]
    common = dict(
        width=int(spec["width"]),
        heads=int(spec["heads"]),
        layers=int(spec["layers"]),
        feedforward_width=int(spec["feedforward_width"]),
        dropout=float(spec["dropout"]),
        maximum_sequence_tokens=int(spec["maximum_sequence_tokens"]),
        maximum_context_bars=int(base["context_policy"]["maximum_context_bars"]),
    )
    if arm == "ARM_E_FULL_COORDINATE":
        return FastCausalUnifiedMatchedMusicModel(_e_config(), **common)
    if arm in TONAL_ARMS:
        return FastCausalUnifiedMatchedMusicModel(_d_config(), **common)
    return FGHIUnifiedMatchedMusicModel(_hi_config(arm), **common)


def _as_fghi(batch: NoteCausalBatch) -> FGHICausalBatch:
    values = {field.name: getattr(batch, field.name) for field in fields(NoteCausalBatch)}
    return FGHICausalBatch(
        **values,
        tonal_frame_pc=torch.zeros_like(batch.input_pitch),
        sample_shift=torch.zeros(
            batch.valid.shape[0], dtype=torch.long, device=batch.valid.device
        ),
    )


def _pitch_bounds(cache: SharedNoteTensorCache, entries: Sequence[Any]) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for entry in entries:
        row = cache.row(entry.sample_id)
        mask = np.asarray(row["kind"]) == INPUT_NOTE
        pitches = np.asarray(row["pitch"])[mask]
        if not len(pitches):
            raise ValueError(f"sample has no notes: {entry.sample_id}")
        result[entry.sample_id] = (int(pitches.min()), int(pitches.max()))
    return result


def _shift_batch(batch: NoteCausalBatch, shifts: torch.Tensor) -> NoteCausalBatch:
    if shifts.shape != (batch.valid.shape[0],):
        raise ValueError("one shift per Pop1K7 batch row is required")
    values = {field.name: getattr(batch, field.name) for field in fields(NoteCausalBatch)}
    delta = shifts[:, None]
    values["input_pitch"] = torch.where(
        batch.input_type == INPUT_NOTE, batch.input_pitch + delta, batch.input_pitch
    )
    values["target_pitch"] = torch.where(
        batch.note_mask, batch.target_pitch + delta, batch.target_pitch
    )
    if bool(
        (
            ((values["input_pitch"] < 0) | (values["input_pitch"] > 127))
            & (batch.input_type == INPUT_NOTE)
        ).any()
    ):
        raise ValueError("shifted input pitch leaves MIDI support")
    if bool(
        (
            ((values["target_pitch"] < 0) | (values["target_pitch"] > 127))
            & batch.note_mask
        ).any()
    ):
        raise ValueError("shifted target pitch leaves MIDI support")
    return NoteCausalBatch(**values)


def prepare_batch(
    batch: NoteCausalBatch,
    entries: Sequence[Any],
    *,
    arm: str,
    seed: int,
    epoch: int,
    domain: str,
    bounds: Mapping[str, tuple[int, int]],
):
    if arm == "ARM_E_FULL_COORDINATE":
        return batch
    if arm in ("ARM_H_FACTORIZED_PITCH_ID", "ARM_I_FIFTHS_CIRCLE_PITCH"):
        return _as_fghi(batch)
    shifts = []
    for entry in entries:
        minimum, maximum = bounds[entry.sample_id]
        shifts.append(
            deterministic_shift(
                entry.sample_id,
                domain=domain,
                epoch=int(epoch),
                seed=int(seed),
                occurrence_index=0,
                minimum_pitch=minimum,
                maximum_pitch=maximum,
                candidates=SHIFT_CANDIDATES,
            )
        )
    shift_tensor = torch.tensor(shifts, dtype=torch.long, device=batch.valid.device)
    if arm == "ARM_F_TRANSPOSED_ABSOLUTE":
        return _shift_batch(batch, shift_tensor)
    # G removes the same known nuisance transform at the tokenizer boundary.
    # The model never receives the shift and sees the canonical C/A fact stream.
    return batch


def _prepared_validation(
    base_batches: Sequence[StaticValidationBatch],
    *,
    arm: str,
    seed: int,
    bounds: Mapping[str, tuple[int, int]],
) -> list[StaticValidationBatch]:
    result = []
    for item in base_batches:
        batch = prepare_batch(
            item.batch,
            item.entries,
            arm=arm,
            seed=seed,
            epoch=-1,
            domain="validation",
            bounds=bounds,
        )
        result.append(
            StaticValidationBatch(
                entries=item.entries,
                batch=batch,
                token_count=item.token_count,
            )
        )
    return result


def run(argv: Sequence[str] | None = None, *, allowed_arms: Sequence[str] = SUPPORTED_ARMS) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(allowed_arms), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--trainer-v2", action="store_true")
    parser.add_argument("--compact-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    protocol_path = args.protocol.resolve()
    protocol, base, base_path = load_remaining_protocol(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    frozen = json.loads(args.protocol_receipt.read_text(encoding="utf-8"))
    if frozen["protocol_sha256"] != protocol_hash or frozen.get("launch_gate_status") != "PASS":
        raise PermissionError("Pop1K7 remaining protocol/gate receipt mismatch")
    if args.arm not in protocol["arms"] or args.seed not in protocol["preregistered_seeds"]:
        raise ValueError("arm or seed is not preregistered")
    if args.physical_gpu_id not in protocol["launch"]["allowed_physical_gpu_ids"]:
        raise PermissionError("GPU is outside Pop1K7 remaining launch set")

    sources = source_tree_receipt(
        remaining_source_paths(protocol_path, base_path), repository_root=ROOT
    )
    if sources["source_tree_sha256"] != frozen["source_tree_sha256"]:
        raise ValueError("Pop1K7 remaining source tree changed after launch gate")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pop1K7 remaining training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = bool(base["training"]["precision"]["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(base["training"]["precision"]["allow_tf32"])
    torch.backends.cudnn.benchmark = bool(base["training"]["cudnn_benchmark"])
    torch.use_deterministic_algorithms(bool(base["training"]["deterministic_algorithms"]))

    manifest = Pop1K7CleanManifest(
        args.manifest.resolve(),
        repository_root=ROOT,
        expected_sha256=protocol["data"]["manifest_sha256"],
        verify_source_files=False,
        defer_test_rows=True,
    )
    train_entries = manifest.train
    validation_entries = manifest.validation
    cache = SharedNoteTensorCache(
        args.tensor_cache.resolve(), expected_manifest_sha256=manifest.manifest_sha256
    )
    expected_cache_sha256 = protocol["data"].get("tensor_cache_manifest_sha256")
    if expected_cache_sha256 and sha256_file(
        args.tensor_cache.resolve() / "cache_manifest.json"
    ) != str(expected_cache_sha256):
        raise ValueError("Pop1K7 remaining tensor cache receipt mismatch")
    if int(cache.metadata["split_counts"].get("test", 0)) != 0:
        raise AssertionError("Pop1K7 note cache contains sealed test rows")
    bounds = _pitch_bounds(cache, (*train_entries, *validation_entries)) if args.arm in TONAL_ARMS else {}

    model = build_model(args.arm, base, seed=args.seed).to(device)
    optimizer = optimizer_for_model(model, base)
    per_epoch = int(base["data"]["train_target_events_per_equivalent_epoch"])
    target_budget = int(base["training"]["target_event_budget"])
    milestones = validation_milestones(base, per_epoch)
    if milestones[-1] != target_budget:
        raise ValueError("Pop1K7 remaining milestone/budget mismatch")

    config_hash = canonical_sha256(protocol["arms"][args.arm])
    run_id = f"pop1k7_{args.arm.lower()}_seed{args.seed}"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identifiers = {
        "run_id": run_id,
        "arm": args.arm,
        "seed": args.seed,
        "protocol_sha256": protocol_hash,
        "config_sha256": config_hash,
        "manifest_sha256": manifest.manifest_sha256,
        "source_tree_sha256": sources["source_tree_sha256"],
        "tensor_cache_manifest_sha256": sha256_file(args.tensor_cache.resolve() / "cache_manifest.json"),
    }
    receipt = {
        "schema_version": "m4l.pop1k7_remaining.run.v1.receipt",
        **identifiers,
        "status": "FORMAL_TRAINING_STARTED",
        "model_parameter_count": parameter_receipt(model),
        "representation_config": protocol["arms"][args.arm],
        "target_event_budget": target_budget,
        "hardware": {**hardware_receipt(device), "declared_physical_gpu_id": args.physical_gpu_id},
        "start_time_utc": utc_now(),
        "end_time_utc": "PENDING",
        "checkpoint_path": "PENDING",
        "test_reads": 0,
        "trainer_v2": bool(args.trainer_v2),
        "compact_checkpoints": bool(args.compact_checkpoints),
    }
    write_json_atomic(output / "run_receipt.json", receipt)
    write_json_atomic(output / "source_tree_hash.json", sources)

    completed = epoch_index = next_batch_index = epoch_exposure = next_milestone_index = 0
    best: dict[str, Any] | None = None
    latest_path = output / "LATEST.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        checkpoint_path = output / latest["checkpoint_path"]
        if sha256_file(checkpoint_path) != latest["sha256"]:
            raise ValueError("Pop1K7 remaining resume checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for name, value in identifiers.items():
            if payload[name] != value:
                raise ValueError(f"Pop1K7 remaining resume {name} mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed = int(payload["completed_target_exposures"])
        epoch_index = int(payload["epoch_index"])
        next_batch_index = int(payload["next_batch_index"])
        epoch_exposure = int(payload["epoch_exposure"])
        next_milestone_index = int(payload["next_milestone_index"])
        best_path = output / "BEST.json"
        best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else payload.get("best")
        restore_rng_state(payload)
        receipt["status"] = "FORMAL_TRAINING_RESUMED"
        write_json_atomic(output / "run_receipt.json", receipt)

    batch_size = int(base["batch_construction"]["batch_songs"])
    bucket_window = int(base["batch_construction"]["length_bucket_window_rows"])
    validation_indices = validation_batches(
        validation_entries,
        batch_songs=int(base["batch_construction"]["validation_batch_songs"]),
    )
    base_validation = build_static_validation_batches(
        cache, validation_entries, validation_indices, pin_memory=True
    )
    static_validation = _prepared_validation(
        base_validation, arm=args.arm, seed=args.seed, bounds=bounds
    )
    history_path = output / "validation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    started = time.perf_counter()

    while completed < target_budget:
        stream_hash = batch_stream_sha256(base, epoch_index)
        indices = epoch_batches(
            train_entries,
            protocol_sha256=stream_hash,
            seed=args.seed,
            epoch_index=epoch_index,
            batch_songs=batch_size,
            bucket_window=bucket_window,
        )
        start_batch = next_batch_index
        entry_batches = [[train_entries[index] for index in row] for row in indices[start_batch:]]
        prefetcher = FastCachedBatchPrefetcher(
            cache,
            entry_batches,
            device=device,
            pin_memory=True,
            workers=1,
            prefetch_depth=2,
        )
        for offset, (prepared, selected) in enumerate(zip(prefetcher, entry_batches)):
            batch_number = start_batch + offset
            batch = prepare_batch(
                prepared.batch,
                selected,
                arm=args.arm,
                seed=args.seed,
                epoch=epoch_index,
                domain="train",
                bounds=bounds,
            )
            batch_count = int(prepared.token_count)
            projected = completed + batch_count
            if projected > target_budget:
                raise AssertionError("Pop1K7 remaining batch exceeds budget")
            lr = learning_rate_for_projected_exposure(base, projected)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, _representation, _hidden = model.losses(batch)
            loss = losses["total"].sum() / max(1, batch_count)
            torch._assert_async(torch.isfinite(loss), "non-finite Pop1K7 remaining loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(base["training"]["optimizer"]["gradient_clip_norm"]),
                error_if_nonfinite=False,
            )
            torch._assert_async(torch.isfinite(gradient_norm), "non-finite Pop1K7 remaining gradient")
            optimizer.step()
            completed = projected
            epoch_exposure += batch_count
            next_batch_index = batch_number + 1
            if next_batch_index == len(indices):
                if epoch_exposure != per_epoch:
                    raise AssertionError("Pop1K7 remaining epoch exposure drift")
                epoch_index += 1
                next_batch_index = 0
                epoch_exposure = 0

            if next_batch_index == 1 or next_batch_index % int(base["training"]["log_every_updates"]) == 0:
                append_jsonl(
                    output / "training_log.jsonl",
                    {
                        "time_utc": utc_now(),
                        "epoch_index": epoch_index,
                        "next_batch_index": next_batch_index,
                        "completed_target_exposures": completed,
                        "target_equivalent_epochs": completed / per_epoch,
                        "loss_nats_per_event": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rate": lr,
                        "batch_events": batch_count,
                        "events_per_second": completed / max(time.perf_counter() - started, 1e-9),
                    },
                )

            if next_milestone_index < len(milestones) and completed >= milestones[next_milestone_index]:
                metrics, per_song = evaluate_cached_v2(
                    model, batches=static_validation, device=device
                )
                milestone = next_milestone_index
                payload = checkpoint_payload(
                    identifiers=identifiers,
                    model=model,
                    optimizer=optimizer,
                    completed=completed,
                    epoch_index=epoch_index,
                    next_batch_index=next_batch_index,
                    epoch_exposure=epoch_exposure,
                    next_milestone_index=milestone + 1,
                    best=best,
                )
                checkpoint_path = output / "checkpoints" / (
                    "latest.pt" if args.compact_checkpoints else f"exposure_{completed:012d}.pt"
                )
                checkpoint_hash = save_checkpoint_atomic(checkpoint_path, payload)
                bits = float(metrics["total_bits_per_event"])
                tie = float(base["validation_and_checkpointing"]["tie_tolerance_bits"])
                improved = best is None or bits < float(best["bits"]) - tie
                if improved:
                    if args.compact_checkpoints:
                        checkpoint_path = output / "checkpoints" / f"best_exposure_{completed:012d}.pt"
                        checkpoint_hash = save_checkpoint_atomic(checkpoint_path, payload)
                    best = {
                        "bits": bits,
                        "completed_target_exposures": completed,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_hash,
                        "milestone_index": milestone,
                    }
                history.append(
                    {
                        "milestone_index": milestone,
                        "scheduled_target_exposure": milestones[milestone],
                        "completed_target_exposures": completed,
                        "target_equivalent_epochs": completed / per_epoch,
                        "checkpoint_path": str(checkpoint_path) if improved or not args.compact_checkpoints else None,
                        "checkpoint_sha256": checkpoint_hash if improved or not args.compact_checkpoints else None,
                        "selected_as_best": improved,
                        "metrics": metrics,
                    }
                )
                write_jsonl_atomic(output / "validation" / f"exposure_{completed:012d}.jsonl", per_song)
                write_json_atomic(history_path, history)
                write_json_atomic(
                    latest_path,
                    {
                        "checkpoint_path": str(checkpoint_path.relative_to(output)),
                        "sha256": checkpoint_hash,
                        "completed_target_exposures": completed,
                    },
                )
                write_json_atomic(output / "BEST.json", best)
                next_milestone_index += 1
                print(
                    json.dumps(
                        {
                            "stage": "pop1k7_remaining_validation",
                            "run_id": run_id,
                            "target_equivalent_epochs": completed / per_epoch,
                            "bits": bits,
                            "best_bits": best["bits"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if next_batch_index != 0:
            raise AssertionError("Pop1K7 remaining epoch rollover incomplete")

    if manifest.test_rows_loaded or manifest.test_reads or manifest.test_row_file_reads:
        raise AssertionError("Pop1K7 remaining training read sealed test")
    if completed != target_budget or best is None or next_milestone_index != len(milestones):
        raise AssertionError("Pop1K7 remaining run incomplete")
    receipt.update(
        {
            "status": "FORMAL_TRAINING_COMPLETE",
            "end_time_utc": utc_now(),
            "checkpoint_path": best["checkpoint_path"],
            "checkpoint_sha256": best["checkpoint_sha256"],
            "completed_target_exposures": completed,
            "selected_validation_bits_per_event": best["bits"],
            "elapsed_seconds": time.perf_counter() - started,
            "test_reads": 0,
        }
    )
    write_json_atomic(output / "run_receipt.json", receipt)
    write_json_atomic(
        output / "report.json",
        {
            "schema_version": "m4l.pop1k7_remaining.run.v1.report",
            "receipt": receipt,
            "validation_history": history,
            "best": best,
        },
    )
    return 0
