from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data.duplicate_families import audit_components_do_not_cross_splits
from .data.pop1k7 import iter_jsonl, verify_official_sources
from .evaluation.factor_decomposition import factor_sum_equals_total
from .evaluation.predictive_ledger import PredictiveLedgerEntry
from .reporting.build_figures import build_figures
from .reporting.build_tables import build_tables
from .reporting.verify_paper_numbers import verify_paper_artifacts
from .representation import build_representation
from .representation.exact_events import decode_exact_events, encode_exact_events
from .representation.reversible_bpe import fit_reversible_bpe
from .representation.serialized_carrier import deserialize_events, serialize_events
from .training.checkpoint_selection import ValidationCandidate, select_checkpoint
from .training.engine import synthetic_train
from .training.test_firewall import TestAccessFirewall


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _artifact_root(value: str | None) -> Path:
    if value:
        return Path(value)
    environment = os.environ.get("ARTIFACT_ROOT")
    return Path(environment) if environment else _repository_root() / "artifacts/paper_v1"


def run_smoke(steps: int = 100) -> dict[str, object]:
    from .data.midi_events import DeclaredEvent

    events = (
        DeclaredEvent("NOTE", 0, 96, 60),
        DeclaredEvent("NOTE", 0, 192, 64),
        DeclaredEvent("REST", 192, 96),
        DeclaredEvent("NOTE", 288, 96, 67),
    )
    exact = decode_exact_events(encode_exact_events(events)) == events
    serialized = serialize_events(events)
    carrier = deserialize_events(serialized) == events
    bpe = fit_reversible_bpe([serialized, serialized], merge_count=8, split="train")
    bpe_round_trip = bpe.decode(bpe.encode(serialized)) == serialized
    ledger = PredictiveLedgerEntry(
        4, 1.0, 2.0, 3.0, 4.0, eos_bits=0.5, side_information_bits=0.25
    )
    training = synthetic_train(steps=steps)
    return {
        "schema_version": "el.synthetic_smoke.v1",
        "status": "PASS"
        if exact
        and carrier
        and bpe_round_trip
        and factor_sum_equals_total(ledger)
        and training.finite
        else "FAIL",
        "scientific_result": False,
        "event_codec_round_trip": exact,
        "serialized_carrier_round_trip": carrier,
        "bpe_round_trip": bpe_round_trip,
        "ledger_factor_sum": factor_sum_equals_total(ledger),
        "synthetic_training": asdict(training),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="el-token")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--steps", type=int, default=100)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--synthetic", action="store_true")
    train.add_argument("--steps", type=int, default=100)

    select = subparsers.add_parser("select")
    select.add_argument("--history", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--validation-lock", type=Path)
    evaluate.add_argument(
        "--index",
        type=Path,
        default=_repository_root() / "artifacts/paper_v1/checkpoint_index.json",
    )

    sealed = subparsers.add_parser("sealed-test")
    sealed.add_argument("--validation-lock", type=Path, required=True)
    sealed.add_argument("--test-manifest", type=Path, required=True)

    paper = subparsers.add_parser("paper")
    paper_sub = paper.add_subparsers(dest="paper_command", required=True)
    for name in ("verify", "tables", "figures"):
        item = paper_sub.add_parser(name)
        item.add_argument("--artifact-root")
        if name != "verify":
            item.add_argument(
                "--output-root", default=str(_repository_root() / "paper/figure_data")
            )

    data = subparsers.add_parser("data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    audit = data_sub.add_parser("audit")
    audit.add_argument("dataset", choices=("pop1k7", "commu"))
    audit.add_argument("--manifest", type=Path, required=True)
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("dataset", choices=("pop1k7", "commu"))
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--source-manifest", type=Path, required=True)

    show = subparsers.add_parser("show-representation")
    show.add_argument("--config", type=Path, required=True)

    args = parser.parse_args(argv)
    result: dict[str, Any]
    if args.command == "smoke":
        result = run_smoke(args.steps)
    elif args.command == "train":
        spec = build_representation(args.config)
        if not args.synthetic:
            raise RuntimeError(
                "real paper training fails closed in release rc1; use scripts/train_arm.py"
            )
        result = {
            "status": "PASS",
            "scientific_result": False,
            "arm": spec.config.arm,
            "training": asdict(synthetic_train(steps=args.steps, seed=args.seed)),
        }
    elif args.command == "select":
        payload = json.loads(args.history.read_text(encoding="utf-8"))
        rows = (
            payload
            if isinstance(payload, list)
            else payload.get("validation_history", payload.get("rows", []))
        )
        selected = select_checkpoint(
            ValidationCandidate(
                path=str(row["checkpoint_path"]),
                bits_per_declared_fact=float(
                    row.get("bits_per_declared_fact", row.get("bits"))
                ),
                completed_target_exposure=int(row["completed_target_exposures"]),
            )
            for row in rows
        )
        result = {"status": "PASS", "test_data_read": False, "selected": asdict(selected)}
    elif args.command == "evaluate":
        spec = build_representation(args.config)
        if args.split == "test" and (
            args.validation_lock is None or not args.validation_lock.is_file()
        ):
            raise PermissionError("test evaluation requires an existing validation lock")
        index = json.loads(args.index.read_text(encoding="utf-8"))
        matches = [
            row
            for row in index["checkpoints"]
            if row["arm"] == spec.config.arm and int(row["seed"]) == args.seed
        ]
        if len(matches) != 1:
            raise ValueError("arm/seed does not identify one indexed checkpoint")
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if digest != matches[0]["checkpoint_sha256"]:
            raise ValueError("checkpoint SHA-256 does not match the frozen index")
        result = {
            "status": "PASS",
            "verification_only": True,
            "checkpoint_id": matches[0]["checkpoint_id"],
            "checkpoint_sha256": digest,
            "split": args.split,
        }
    elif args.command == "sealed-test":
        if not args.validation_lock.is_file():
            raise FileNotFoundError(args.validation_lock)
        firewall = TestAccessFirewall()
        firewall.lock_validation()
        rows = firewall.read_test_rows_once(args.test_manifest)
        result = {
            "status": "PASS",
            "enforceable_security_boundary": False,
            "rows_read": len(rows),
            "test_manifest_accessor_calls": firewall.accessor_calls,
            "test_row_file_reads": firewall.row_file_reads,
            "checkpoint_updates": 0,
            "test_time_bpe_fit_updates": 0,
        }
    elif args.command == "paper":
        artifact_root = _artifact_root(args.artifact_root)
        if args.paper_command == "verify":
            result = verify_paper_artifacts(artifact_root)
        elif args.paper_command == "tables":
            result = build_tables(artifact_root, Path(args.output_root))
        else:
            result = build_figures(artifact_root, Path(args.output_root))
    elif args.command == "data" and args.data_command == "audit":
        rows = list(iter_jsonl(args.manifest))
        crossings = audit_components_do_not_cross_splits(rows)
        required_fields = (
            (
                "lineage_id",
                "exact_hash",
                "transposition_invariant_hash",
            )
            if args.dataset == "commu"
            else (
                "lineage_id",
                "exact_hash",
                "transposition_invariant_hash",
                "rhythm_interval_hash",
            )
        )
        present_fields = {
            field for row in rows for field, value in row.items() if value is not None
        }
        missing_required_fields = [
            field for field in required_fields if field not in present_fields
        ]
        result = {
            "status": "PASS"
            if not missing_required_fields
            and all(crossings[field] == 0 for field in required_fields)
            else "FAIL",
            "dataset": args.dataset,
            "rows": len(rows),
            "required_zero_crossing_fields": list(required_fields),
            "missing_required_fields": missing_required_fields,
            "cross_split_component_counts": crossings,
        }
    elif args.command == "data":
        verification = verify_official_sources(args.root, args.source_manifest)
        result = {
            **verification,
            "status": "PASS"
            if verification["missing"] == 0 and verification["mismatched"] == 0
            else "FAIL",
            "dataset": args.dataset,
        }
    else:
        result = build_representation(args.config).receipt()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
