#!/usr/bin/env python3
"""Public training entrypoint with an explicit synthetic/real-run boundary."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from el_tokenization.representation import build_representation
from el_tokenization.training.engine import synthetic_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    spec = build_representation(args.config)
    if args.synthetic:
        result = synthetic_train(steps=args.steps, seed=args.seed)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scientific_result": False,
                    "arm": spec.config.arm,
                    "training": asdict(result),
                },
                indent=2,
            )
        )
        return
    if args.prepared_root is None or not args.prepared_root.is_dir():
        raise ValueError("a verified --prepared-root is required for paper training")
    raise RuntimeError(
        "the clean-room full-training adapter is not certified in release rc1; "
        "paper-compatible low-level modules and frozen protocols are included, but this command fails closed"
    )


if __name__ == "__main__":
    main()
