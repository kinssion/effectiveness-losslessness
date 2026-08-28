from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    materialized = list(rows)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in materialized
        ),
        encoding="utf-8",
        newline="\n",
    )
    return len(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pop1k7_relative(value: object) -> str:
    path = str(value).replace("\\", "/")
    prefix = "data/pop1k7_official/Pop1K7/"
    return path.removeprefix(prefix)


def commu_relative(value: object) -> str:
    path = str(value).replace("\\", "/")
    prefix = "data/commu_official/dataset/"
    return path.removeprefix(prefix)


def build_pop1k7() -> dict[str, int]:
    root = ROOT / "manifests/pop1k7"
    frozen = root / "frozen_split_v1"
    songs = read_jsonl(frozen / "songs.jsonl")
    source_rows = [
        {
            "song_id": row["source_song_id"],
            "relative_source_path": pop1k7_relative(row["source_path"]),
            "content_sha256": row["source_byte_sha256"],
            "duplicate_component_id": row["lineage_id"],
            "split": row["split"],
            "num_bars": row["bar_count"],
            "time_signature": "4/4",
        }
        for row in songs
    ]
    duplicate_rows = [
        {
            "song_id": row["source_song_id"],
            "duplicate_component_id": row["lineage_id"],
            "lineage_id": row["lineage_id"],
            "lineage_status": row["lineage_status"],
            "exact_hash": row["whole_exact_hash"],
            "transposition_invariant_hash": row["whole_transposition_invariant_hash"],
            "rhythm_interval_hash": row["whole_rhythm_interval_hash"],
            "split": row["split"],
        }
        for row in songs
    ]
    split_rows = [
        {
            "song_id": row["source_song_id"],
            "duplicate_component_id": row["lineage_id"],
            "split": row["split"],
        }
        for row in songs
    ]
    windows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        for row in read_jsonl(frozen / f"{split}.jsonl"):
            row["source_path"] = pop1k7_relative(row["source_path"])
            windows.append(row)

    high_confidence = read_jsonl(root / "frozen_high_confidence_windows_v1/songs.jsonl")
    for row in high_confidence:
        row["source_path"] = pop1k7_relative(row["source_path"])

    counts = {
        "source_manifest.jsonl": write_jsonl(root / "source_manifest.jsonl", source_rows),
        "duplicate_components.jsonl": write_jsonl(
            root / "duplicate_components.jsonl", duplicate_rows
        ),
        "song_split_v1.jsonl": write_jsonl(root / "song_split_v1.jsonl", split_rows),
        "window_manifest_v1.jsonl": write_jsonl(root / "window_manifest_v1.jsonl", windows),
        "high_confidence_subset_v1.jsonl": write_jsonl(
            root / "high_confidence_subset_v1.jsonl", high_confidence
        ),
    }

    split_by_song = {str(row["source_song_id"]): str(row["split"]) for row in songs}
    estimates = read_jsonl(root / "frozen_key_estimation_v1/key_estimates.jsonl")
    csv_path = root / "key_estimation_v1.csv"
    fields = [
        "song_id",
        "content_sha256",
        "analyzer_1_key",
        "analyzer_2_key",
        "agreement",
        "confidence",
        "accepted",
        "canonical_shift",
        "split",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in estimates:
            analyses = cast(dict[str, Any], row.get("analyses", {}))
            first = analyses.get("krumhansl_schmuckler", {})
            second = analyses.get("aarden_essen", {})
            first_margin = float(first.get("distinct_frame_margin", 0.0))
            second_margin = float(second.get("distinct_frame_margin", 0.0))
            writer.writerow(
                {
                    "song_id": row["source_song_id"],
                    "content_sha256": row["model_source_sha256"],
                    "analyzer_1_key": first.get("label"),
                    "analyzer_2_key": second.get("label"),
                    "agreement": bool(row.get("same_relative_major_frame")),
                    "confidence": min(first_margin, second_margin),
                    "accepted": row.get("status") == "ADMITTED_HIGH_CONFIDENCE",
                    "canonical_shift": row.get("canonical_shift"),
                    "split": split_by_song.get(str(row["source_song_id"]), "excluded"),
                }
            )
    counts[csv_path.name] = len(estimates)
    return counts


def build_commu() -> dict[str, int]:
    root = ROOT / "manifests/commu"
    frozen = root / "frozen_split_v1"
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        rows.extend(read_jsonl(frozen / f"{split}.jsonl"))
    source_rows = [
        {
            "song_id": row["source_song_id"],
            "relative_source_path": commu_relative(row["source_path"]),
            "content_sha256": None,
            "exact_event_sha256": row["exact_hash"],
            "duplicate_component_id": row["lineage_id"],
            "split": row["split"],
            "num_bars": row["num_measures"],
            "time_signature": row["time_signature"],
        }
        for row in rows
    ]
    duplicate_rows = [
        {
            "song_id": row["source_song_id"],
            "duplicate_component_id": row["lineage_id"],
            "lineage_id": row["lineage_id"],
            "exact_hash": row["exact_hash"],
            "transposition_invariant_hash": row["transposition_invariant_hash"],
            "rhythm_interval_hash": row["rhythm_interval_hash"],
            "split": row["split"],
        }
        for row in rows
    ]
    return {
        "source_manifest.jsonl": write_jsonl(root / "source_manifest.jsonl", source_rows),
        "duplicate_components.jsonl": write_jsonl(
            root / "duplicate_components.jsonl", duplicate_rows
        ),
        "song_split_v1.jsonl": write_jsonl(
            root / "song_split_v1.jsonl",
            ({"song_id": row["source_song_id"], "split": row["split"]} for row in rows),
        ),
    }


def main() -> None:
    counts = {"pop1k7": build_pop1k7(), "commu": build_commu()}
    generated = [
        path
        for path in (ROOT / "manifests").rglob("*")
        if path.is_file() and "frozen_" not in path.as_posix()
    ]
    receipt = {
        "schema_version": "el.public_manifest_build.v1",
        "status": "PASS",
        "counts": counts,
        "files": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in sorted(generated)
            if path.name != "manifest_build_receipt.json"
        },
    }
    path = ROOT / "manifests/manifest_build_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
