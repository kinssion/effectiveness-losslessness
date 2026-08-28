#!/usr/bin/env python3
"""Select a validation checkpoint without reading any test result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from el_tokenization.training.checkpoint_selection import ValidationCandidate, select_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.history.read_text(encoding="utf-8"))
    rows = (
        payload
        if isinstance(payload, list)
        else payload.get("validation_history", payload.get("rows", []))
    )
    candidates = [
        ValidationCandidate(
            path=str(row["checkpoint_path"]),
            bits_per_declared_fact=float(row.get("bits_per_declared_fact", row.get("bits"))),
            completed_target_exposure=int(row["completed_target_exposures"]),
        )
        for row in rows
    ]
    selected = select_checkpoint(candidates)
    receipt = {
        "schema_version": "el.validation_selection.v1",
        "test_data_read": False,
        "selected_checkpoint": selected.path,
        "validation_bits_per_declared_fact": selected.bits_per_declared_fact,
        "completed_target_exposure": selected.completed_target_exposure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
