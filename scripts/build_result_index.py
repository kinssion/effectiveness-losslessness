#!/usr/bin/env python3
"""Build checkpoint and result indices from released frozen ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_rows(root: Path) -> list[dict[str, Any]]:
    aggregate = root / "aggregate_results"
    ai = _load(aggregate / "pop1k7_ai_paper_ledger.json")
    jk = _load(aggregate / "pop1k7_jk_paper_ledger.json")
    commu = _load(aggregate / "commu_ad_paper_ledger.json")
    rows: list[dict[str, Any]] = []
    for family, result, dataset in (
        ("pop1k7_ai", ai, "pop1k7"),
        ("pop1k7_jk", jk, "pop1k7"),
        ("commu_ad", commu, "commu"),
    ):
        for item in result["per_run"]:
            arm = str(item["arm"])
            if family == "pop1k7_jk" and arm == "D":
                continue
            seed = int(item["seed"])
            rows.append(
                {
                    "checkpoint_id": f"{dataset}_{arm}_seed{seed}",
                    "family": family,
                    "dataset": dataset,
                    "arm": arm,
                    "seed": seed,
                    "config": f"configs/{dataset}/{arm}.yaml",
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "expected_filename": f"{dataset}_{arm}_seed{seed}.safetensors",
                    "download_url": None,
                    "availability": "not_bundled_pending_upstream_license_review",
                    "weight_license": "pending_upstream_review",
                }
            )
    rows.append(
        {
            "checkpoint_id": "context_probe_seed20260801",
            "family": "context_probe",
            "dataset": "derived_context_features",
            "arm": "same_song_relation_probe",
            "seed": 20260801,
            "config": "configs/context_probe/context_relation.yaml",
            "checkpoint_sha256": "9a7475f975c5f100896b1452d50df8544a37d3e400060d9f20eb6d6093e60260",
            "expected_filename": "context_probe_seed20260801.safetensors",
            "download_url": None,
            "availability": "not_bundled_pending_upstream_license_review",
            "weight_license": "pending_upstream_review",
        }
    )
    return sorted(rows, key=lambda row: str(row["checkpoint_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/paper_v1"))
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    checkpoints = _checkpoint_rows(root)
    checkpoint_index = {
        "schema_version": "el.checkpoint_index.v1",
        "weight_files_in_git": False,
        "publication_policy": "separate immutable release after upstream-license review",
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }
    _dump(root / "checkpoint_index.json", checkpoint_index)

    excluded = {"result_index.json", "hashes/result_index.sha256"}
    files = {
        path.relative_to(root).as_posix(): _hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }
    result_index = {
        "schema_version": "el.result_index.v1",
        "review_artifact": True,
        "files": files,
        "claims": {
            "pop1k7_ai": {
                "result": "aggregate_results/pop1k7_ai_paper_ledger.json",
                "configs": [f"configs/pop1k7/{arm}.yaml" for arm in "ABCDEFGHI"],
                "checkpoint_ids": [
                    row["checkpoint_id"] for row in checkpoints if row["family"] == "pop1k7_ai"
                ],
                "means": {
                    "A": 7.1161798815495745,
                    "B": 6.4222314569972685,
                    "C": 6.326624368934327,
                    "D": 6.2680929492472295,
                    "E": 6.278414314185332,
                    "F": 6.486857454991204,
                    "G": 6.396309754810838,
                    "H": 6.23638194766388,
                    "I": 6.334688793264417,
                },
            },
            "pop1k7_jk": {
                "result": "aggregate_results/pop1k7_jk_paper_ledger.json",
                "configs": ["configs/pop1k7/J.yaml", "configs/pop1k7/K.yaml"],
                "checkpoint_ids": [
                    row["checkpoint_id"] for row in checkpoints if row["family"] == "pop1k7_jk"
                ],
                "means": {"J": 5.963053681364638, "K": 6.773668209606095},
            },
            "commu_ad": {
                "result": "aggregate_results/commu_ad_paper_ledger.json",
                "configs": ["configs/commu/A.yaml", "configs/commu/D.yaml"],
                "checkpoint_ids": [
                    row["checkpoint_id"] for row in checkpoints if row["family"] == "commu_ad"
                ],
                "means": {"A": 12.339934429973125, "D": 12.228547559219846},
            },
            "context_probe": {
                "result": "context_probe/context_dependence_audit.json",
                "checkpoint_ids": ["context_probe_seed20260801"],
                "manifest_status": "historical held-out window manifest not recovered",
            },
        },
    }
    _dump(root / "result_index.json", result_index)
    index_hash = _hash(root / "result_index.json")
    hash_path = root / "hashes/result_index.sha256"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(f"{index_hash}  result_index.json\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": "PASS", "indexed_files": len(files), "checkpoints": len(checkpoints)}
        )
    )


if __name__ == "__main__":
    main()
