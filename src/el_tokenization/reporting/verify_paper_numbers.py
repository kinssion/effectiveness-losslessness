from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..evaluation.receipts import file_sha256, validate_receipt_fields


def verify_paper_artifacts(artifact_root: Path) -> dict[str, object]:
    index_path = artifact_root / "result_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    hash_failures: list[str] = []
    for relative, expected in index["files"].items():
        path = artifact_root / relative
        if not path.is_file() or file_sha256(path) != expected:
            hash_failures.append(relative)

    number_failures: list[str] = []
    loaded_claims: dict[str, Any] = {}
    for claim_name in ("pop1k7_ai", "pop1k7_jk", "commu_ad"):
        claim = index["claims"][claim_name]
        result = json.loads((artifact_root / claim["result"]).read_text(encoding="utf-8"))
        loaded_claims[claim_name] = result
        by_arm = {
            row["arm"]: float(row["mean_bits_per_declared_fact"])
            for row in result["arm_summary"]
        }
        number_failures.extend(
            f"{claim_name}:{arm}"
            for arm, expected in claim["means"].items()
            if arm not in by_arm or abs(by_arm[arm] - float(expected)) > 1e-12
        )

    required_receipt_fields = (
        "status",
        "test_manifest_accessor_calls",
        "test_row_file_reads",
        "checkpoint_updates",
        "test_time_bpe_fit_updates",
    )
    jk = loaded_claims["pop1k7_jk"]
    receipt_missing = validate_receipt_fields(jk["test_access"], required_receipt_fields)
    access = jk["test_access"]
    access_failures = [
        field
        for field, expected in {
            "test_manifest_accessor_calls": 1,
            "test_row_file_reads": 1,
            "checkpoint_updates": 0,
            "test_time_bpe_fit_updates": 0,
        }.items()
        if access.get(field) != expected
    ]
    jk_count_failures = [
        f"{row['arm']}:{row['seed']}"
        for row in jk["per_run"]
        if row["evaluated_window_count"] != 1979
        or row["model_native_target_count_including_eos"] != 244248
        or row["declared_fact_count"] != 242269
    ]
    checkpoint_index = json.loads(
        (artifact_root / "checkpoint_index.json").read_text(encoding="utf-8")
    )
    checkpoint_failures = [
        row["checkpoint_id"]
        for row in checkpoint_index["checkpoints"]
        if len(str(row["checkpoint_sha256"])) != 64
    ]
    status = (
        "PASS"
        if not (
            hash_failures
            or number_failures
            or receipt_missing
            or access_failures
            or jk_count_failures
            or checkpoint_failures
            or checkpoint_index.get("checkpoint_count") != 40
        )
        else "FAIL"
    )
    return {
        "schema_version": "el.paper_artifact_verification.v1",
        "status": status,
        "indexed_file_count": len(index["files"]),
        "hash_failures": hash_failures,
        "paper_number_failures": number_failures,
        "receipt_missing_fields": list(receipt_missing),
        "test_access_failures": access_failures,
        "jk_denominator_failures": jk_count_failures,
        "checkpoint_index_count": checkpoint_index.get("checkpoint_count"),
        "checkpoint_hash_format_failures": checkpoint_failures,
    }
