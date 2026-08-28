#!/usr/bin/env python3
"""Derive the paper ledger from the frozen, model-native clean-test results.

No number in this script is a measured result.  Counts and native code lengths
come from the released JSON files; the only protocol constant is the four-bit
per-window inverse shift code for Pop1K7 arm G.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(rows: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for arm in sorted({str(row["arm"]) for row in rows}):
        selected = sorted(
            (row for row in rows if row["arm"] == arm), key=lambda row: int(row["seed"])
        )
        values = [float(row[value_key]) for row in selected]
        first = selected[0]
        output.append(
            {
                "arm": arm,
                "declared_fact_count": int(first["declared_fact_count"]),
                "eos_target_count": int(first["eos_target_count"]),
                "mean_bits_per_declared_fact": mean(values),
                "sample_sd_bits_per_declared_fact": stdev(values),
                "seed_values": {str(row["seed"]): row[value_key] for row in selected},
                "side_information_bits_per_window": first.get(
                    "side_information_bits_per_window", 0
                ),
                "serialized_targets_per_declared_fact": first.get(
                    "serialized_targets_per_declared_fact"
                ),
            }
        )
    return output


def _paired(
    rows: list[dict[str, Any]], comparisons: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["arm"]), int(row["seed"])): float(row["paper_bits_per_declared_fact"])
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        seeds = sorted(
            {seed for arm, seed in by_key if arm == left}
            & {seed for arm, seed in by_key if arm == right}
        )
        deltas = {str(seed): by_key[(left, seed)] - by_key[(right, seed)] for seed in seeds}
        values = list(deltas.values())
        output.append(
            {
                "comparison": f"{left}-{right}",
                "left_arm": left,
                "right_arm": right,
                "mean_delta_bits_per_declared_fact": mean(values),
                "sample_sd_delta_bits_per_declared_fact": stdev(values),
                "all_seeds_same_direction": all(value > 0 for value in values)
                or all(value < 0 for value in values),
                "seed_deltas": deltas,
            }
        )
    return output


def build_pop1k7_ai(source: Path, destination: Path, factor_csv: Path) -> None:
    raw = _load(source)
    rows: list[dict[str, Any]] = []
    component_names = ("type", "time", "pitch", "duration")
    for source_row in raw["per_run"]:
        event_count = int(source_row["evaluated_event_count"])
        declared_count = int(source_row["evaluated_support_event_count"])
        eos_count = event_count - declared_count
        arm = str(source_row["arm"])
        factor = event_count / declared_count
        side_bits_per_window = 4 if arm == "G" else 0
        side_bits = side_bits_per_window * int(source_row["evaluated_song_count"])
        components = {
            f"{name}_bits_per_declared_fact": float(
                source_row[f"{name}_bits_per_event_contribution"]
            )
            * factor
            for name in component_names
        }
        model_bits = float(source_row["test_bits_per_exact_event"]) * event_count
        paper_total = (model_bits + side_bits) / declared_count
        row = {
            "arm": arm,
            "seed": int(source_row["seed"]),
            "checkpoint_sha256": source_row["checkpoint_sha256"],
            "evaluated_window_count": int(source_row["evaluated_song_count"]),
            "model_native_target_count_including_eos": event_count,
            "declared_fact_count": declared_count,
            "eos_target_count": eos_count,
            "model_native_bits_per_target": float(source_row["test_bits_per_exact_event"]),
            "model_native_numerator_bits_including_eos": model_bits,
            "side_information_bits_per_window": side_bits_per_window,
            "side_information_bits": side_bits,
            "paper_bits_per_declared_fact": paper_total,
            "eos_bits_reporting": "included in type_bits; the frozen evaluator did not log EOS NLL separately",
            **components,
        }
        row["factor_sum_bits_per_declared_fact"] = (
            sum(components.values()) + side_bits / declared_count
        )
        rows.append(row)

    result = {
        "schema_version": "el.paper_ledger.pop1k7_ai.v1",
        "source_result": source.name,
        "source_result_sha256": _sha256(source),
        "metric": "base-2 predictive code length per declared Note/REST fact",
        "accounting": {
            "numerator": "all model target bits including one EOS per evaluated window, plus required side information",
            "denominator": "declared Note/REST facts only",
            "arm_G_side_code": "ceil(log2(12)) = 4 bits per independently canonicalized window",
            "separate_EOS_NLL": "not present in the historical frozen evaluator; EOS bits are included in the type channel",
        },
        "arm_summary": _summary(rows, "paper_bits_per_declared_fact"),
        "paired_comparisons": _paired(
            rows,
            [
                ("B", "A"),
                ("C", "B"),
                ("D", "C"),
                ("D", "A"),
                ("H", "D"),
                ("E", "D"),
                ("I", "H"),
                ("G", "F"),
            ],
        ),
        "per_run": sorted(rows, key=lambda row: (str(row["arm"]), int(row["seed"]))),
    }
    _dump(destination, result)

    factor_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "seed",
        "evaluated_window_count",
        "model_native_target_count_including_eos",
        "declared_fact_count",
        "eos_target_count",
        "type_bits_per_declared_fact",
        "time_bits_per_declared_fact",
        "pitch_bits_per_declared_fact",
        "duration_bits_per_declared_fact",
        "side_information_bits_per_window",
        "side_information_bits",
        "factor_sum_bits_per_declared_fact",
        "paper_bits_per_declared_fact",
    ]
    with factor_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in fields}
            for row in sorted(rows, key=lambda value: (str(value["arm"]), int(value["seed"])))
        )


def build_pop1k7_jk(source: Path, destination: Path) -> None:
    raw = _load(source)
    rows: list[dict[str, Any]] = []
    for source_row in raw["per_run"]:
        event_count = int(source_row["evaluated_original_event_count"])
        eos_count = int(source_row["evaluated_song_count"])
        declared_count = event_count - eos_count
        serialized_count = source_row.get("evaluated_serialized_target_count")
        rows.append(
            {
                "arm": str(source_row["arm"]),
                "seed": int(source_row["seed"]),
                "checkpoint_sha256": source_row["checkpoint_sha256"],
                "evaluated_window_count": eos_count,
                "model_native_target_count_including_eos": event_count,
                "declared_fact_count": declared_count,
                "eos_target_count": eos_count,
                "model_native_bits_per_original_event_or_eos": float(
                    source_row["test_bits_per_exact_event"]
                ),
                "paper_bits_per_declared_fact": float(source_row["test_bits_per_exact_event"])
                * event_count
                / declared_count,
                "serialized_target_count_including_eos": serialized_count,
                "serialized_targets_per_declared_fact": None
                if serialized_count is None
                else int(serialized_count) / declared_count,
                "side_information_bits_per_window": 0,
                "eos_bits_reporting": "included in the complete arm-specific numerator; not separately logged",
            }
        )
    result = {
        "schema_version": "el.paper_ledger.pop1k7_jk.v1",
        "source_result": source.name,
        "source_result_sha256": _sha256(source),
        "metric": "base-2 predictive code length per declared Note/REST fact",
        "accounting": {
            "numerator": "complete arm-specific numerator including one EOS per evaluated window",
            "denominator": "declared Note/REST facts only",
            "technical_continuation": "used only the verified frozen cache; no source MIDI or test manifest reopen",
        },
        "arm_summary": _summary(rows, "paper_bits_per_declared_fact"),
        "paired_comparisons": _paired(rows, [("K", "J"), ("J", "D"), ("K", "D")]),
        "test_access": {
            "status": raw["status"],
            "test_manifest_accessor_calls": raw["test_manifest_accessor_calls"],
            "test_row_file_reads": raw["test_row_file_reads"],
            "checkpoint_updates": raw["checkpoint_updates"],
            "test_time_bpe_fit_updates": raw["test_time_bpe_fit_updates"],
        },
        "per_run": sorted(rows, key=lambda row: (str(row["arm"]), int(row["seed"]))),
    }
    _dump(destination, result)


def build_commu(source: Path, destination: Path) -> None:
    raw = _load(source)
    rows: list[dict[str, Any]] = []
    for source_row in raw["runs"]:
        metrics = source_row["clean_test_metrics"]
        event_count = int(metrics["evaluated_event_count"])
        eos_count = int(metrics["evaluated_song_count"])
        declared_count = event_count - eos_count
        arm = "A" if source_row["arm"] == "ARM_A_RAW_SEQUENCE" else "D"
        rows.append(
            {
                "arm": arm,
                "seed": int(source_row["seed"]),
                "checkpoint_sha256": source_row["checkpoint_sha256"],
                "evaluated_window_count": eos_count,
                "model_native_target_count_including_eos": event_count,
                "declared_fact_count": declared_count,
                "eos_target_count": eos_count,
                "model_native_bits_per_target": float(metrics["total_bits_per_event"]),
                "paper_bits_per_declared_fact": float(metrics["total_bits_per_event"])
                * event_count
                / declared_count,
                "side_information_bits_per_window": 0,
                "eos_bits_reporting": "included in the model numerator; not separately logged",
            }
        )
    result = {
        "schema_version": "el.paper_ledger.commu_ad.v1",
        "source_result": source.name,
        "source_result_sha256": _sha256(source),
        "metric": "base-2 predictive code length per declared Note/REST fact",
        "accounting": {
            "numerator": "all model target bits including one EOS per evaluated sequence",
            "denominator": "declared Note/REST facts only",
        },
        "arm_summary": _summary(rows, "paper_bits_per_declared_fact"),
        "paired_comparisons": _paired(rows, [("D", "A")]),
        "per_run": sorted(rows, key=lambda row: (str(row["arm"]), int(row["seed"]))),
    }
    _dump(destination, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/paper_v1"))
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    aggregate = root / "aggregate_results"
    build_pop1k7_ai(
        aggregate / "pop1k7_ai_clean_test_results.json",
        aggregate / "pop1k7_ai_paper_ledger.json",
        root / "factor_results/pop1k7_ai_component_paper_ledger_by_seed.csv",
    )
    build_pop1k7_jk(
        aggregate / "pop1k7_jk_clean_test_results.json",
        aggregate / "pop1k7_jk_paper_ledger.json",
    )
    build_commu(
        aggregate / "commu_ad_clean_test_results.json",
        aggregate / "commu_ad_paper_ledger.json",
    )
    print(json.dumps({"status": "PASS", "artifact_root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
