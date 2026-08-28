#!/usr/bin/env python3
"""Fail-closed checkpoint gate for the separately published evaluation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from el_tokenization.representation import build_representation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-lock", type=Path)
    parser.add_argument(
        "--index", type=Path, default=Path("artifacts/paper_v1/checkpoint_index.json")
    )
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    spec = build_representation(args.config)
    if args.split == "test" and (
        args.validation_lock is None or not args.validation_lock.is_file()
    ):
        raise PermissionError("test evaluation requires an existing validation lock")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    candidates = [
        row
        for row in index["checkpoints"]
        if row["arm"] == spec.config.arm and int(row["seed"]) == args.seed
    ]
    if len(candidates) != 1:
        raise ValueError("arm/seed does not identify one indexed checkpoint")
    selected = candidates[0]
    observed = _sha256(args.checkpoint)
    if observed != selected["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match the frozen index")
    receipt = {
        "status": "HASH_VERIFIED",
        "checkpoint_id": selected["checkpoint_id"],
        "checkpoint_sha256": observed,
        "arm": spec.config.arm,
        "seed": args.seed,
        "split": args.split,
        "active_parameter_count": spec.active_parameter_count,
    }
    if not args.verify_only:
        if args.prepared_root is None:
            raise ValueError("--prepared-root is required for model evaluation")
        raise RuntimeError(
            "clean-room checkpoint evaluator adapter is gated until the separate weight bundle is published; "
            "use --verify-only to validate the release object"
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
