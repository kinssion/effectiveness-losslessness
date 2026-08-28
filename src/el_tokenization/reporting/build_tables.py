from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(artifact_root: Path, output_root: Path) -> dict[str, str]:
    ai = _load(artifact_root / "aggregate_results/pop1k7_ai_paper_ledger.json")
    jk = _load(artifact_root / "aggregate_results/pop1k7_jk_paper_ledger.json")
    commu = _load(artifact_root / "aggregate_results/commu_ad_paper_ledger.json")

    ai_rows = [
        {
            "arm": row["arm"],
            "mean_bits_per_declared_fact": row["mean_bits_per_declared_fact"],
            "sample_sd": row["sample_sd_bits_per_declared_fact"],
        }
        for row in ai["arm_summary"]
    ]
    _write_csv(
        output_root / "table_pop1k7_ai.csv",
        ["arm", "mean_bits_per_declared_fact", "sample_sd"],
        ai_rows,
    )

    jk_rows: list[dict[str, Any]] = []
    for row in jk["arm_summary"]:
        jk_rows.append(
            {
                "arm": row["arm"],
                "mean_bits_per_declared_fact": row["mean_bits_per_declared_fact"],
                "sample_sd": row["sample_sd_bits_per_declared_fact"],
                "serialized_targets_per_declared_fact": row.get(
                    "serialized_targets_per_declared_fact"
                ),
            }
        )
    _write_csv(
        output_root / "table_pop1k7_jk.csv",
        [
            "arm",
            "mean_bits_per_declared_fact",
            "sample_sd",
            "serialized_targets_per_declared_fact",
        ],
        jk_rows,
    )

    commu_rows = [
        {
            "arm": row["arm"],
            "mean_bits_per_declared_fact": row["mean_bits_per_declared_fact"],
            "sample_sd": row["sample_sd_bits_per_declared_fact"],
        }
        for row in commu["arm_summary"]
    ]
    _write_csv(
        output_root / "table_commu_ad.csv",
        ["arm", "mean_bits_per_declared_fact", "sample_sd"],
        commu_rows,
    )

    markdown = [
        "# Reconstructed paper tables",
        "",
        "All values below were read from released frozen JSON artifacts.",
        "",
        "## Pop1K7 A-I",
        "",
        "| Arm | bits/declared fact | sample SD |",
        "|---|---:|---:|",
    ]
    markdown.extend(
        f"| {row['arm']} | {float(row['mean_bits_per_declared_fact']):.5f} | {float(row['sample_sd']):.5f} |"
        for row in ai_rows
    )
    markdown.extend(
        ["", "## Pop1K7 J/K", "", "| Arm | bits/declared fact | sample SD |", "|---|---:|---:|"]
    )
    markdown.extend(
        f"| {row['arm']} | {float(row['mean_bits_per_declared_fact']):.5f} | {float(row['sample_sd']):.5f} |"
        for row in jk_rows
    )
    markdown.extend(
        ["", "## ComMU A/D", "", "| Arm | bits/declared fact | sample SD |", "|---|---:|---:|"]
    )
    markdown.extend(
        f"| {row['arm']} | {float(row['mean_bits_per_declared_fact']):.5f} | {float(row['sample_sd']):.5f} |"
        for row in sorted(commu_rows, key=lambda value: str(value["arm"]))
    )
    markdown_path = output_root / "paper_tables.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        "pop1k7_ai": str(output_root / "table_pop1k7_ai.csv"),
        "pop1k7_jk": str(output_root / "table_pop1k7_jk.csv"),
        "commu_ad": str(output_root / "table_commu_ad.csv"),
        "markdown": str(markdown_path),
    }
