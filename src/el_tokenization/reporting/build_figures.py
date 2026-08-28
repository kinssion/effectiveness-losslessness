from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

STYLE = (
    "<style>.title{font:600 14px sans-serif}.label,.value,.axislabel{font:12px sans-serif}"
    ".axis{stroke:#555}.grid{stroke:#d9e2ec}.line{fill:none;stroke:#2864b7;stroke-width:2}"
    ".thin{fill:none;stroke-width:1;opacity:.34}.mean{fill:none;stroke-width:2.2}"
    ".point{fill:#2864b7}.error{stroke:#667085;stroke-width:1.3}</style>"
)


def _panel(
    title: str,
    values: list[tuple[str, float, float]],
    x0: int,
    width: int = 280,
) -> str:
    x_left, x_right, y_bottom, height = x0 + 30, x0 + width - 25, 235, 145
    lower = min(value - error for _, value, error in values) - 0.015
    upper = max(value + error for _, value, error in values) + 0.015

    def y_position(value: float) -> int:
        return y_bottom - round((value - lower) * height / (upper - lower))

    plotted: list[tuple[str, int, int, float, int, int]] = []
    for index, (label, value, error) in enumerate(values):
        x = x_left + (
            0 if len(values) == 1 else round(index * (x_right - x_left) / (len(values) - 1))
        )
        plotted.append(
            (
                label,
                x,
                y_position(value),
                value,
                y_position(value + error),
                y_position(value - error),
            )
        )
    polyline = " ".join(f"{x},{y}" for _, x, y, _, _, _ in plotted)
    elements = [
        f'<text x="{x0 + width / 2}" y="30" text-anchor="middle" class="title">{html.escape(title)}</text>',
        f'<line x1="{x_left}" y1="{y_bottom}" x2="{x_right}" y2="{y_bottom}" class="axis"/>',
        f'<polyline points="{polyline}" class="line"/>',
    ]
    for label, x, y, value, error_top, error_bottom in plotted:
        elements.extend(
            (
                f'<line x1="{x}" y1="{error_top}" x2="{x}" y2="{error_bottom}" class="error"/>',
                f'<line x1="{x - 4}" y1="{error_top}" x2="{x + 4}" y2="{error_top}" class="error"/>',
                f'<line x1="{x - 4}" y1="{error_bottom}" x2="{x + 4}" y2="{error_bottom}" class="error"/>',
                f'<circle cx="{x}" cy="{y}" r="5" class="point"/>',
                f'<text x="{x}" y="{y - 10}" text-anchor="middle" class="value">{value:.3f}</text>',
                f'<text x="{x}" y="255" text-anchor="middle" class="label">{html.escape(label)}</text>',
            )
        )
    return "\n".join(elements)


def _curve_figure(artifact_root: Path, output_root: Path) -> Path:
    long_path = artifact_root / "learning_curves/pop1k7_ai_validation_long.csv"
    mean_path = artifact_root / "learning_curves/pop1k7_ai_validation_mean_sd.csv"
    with long_path.open(encoding="utf-8", newline="") as handle:
        long_rows = [
            row
            for row in csv.DictReader(handle)
            if float(row["target_equivalent_epochs"]) >= 100.0
        ]
    with mean_path.open(encoding="utf-8", newline="") as handle:
        mean_rows = [
            row
            for row in csv.DictReader(handle)
            if float(row["mean_target_equivalent_epochs"]) >= 100.0
        ]

    panels = [
        ("A-D temporal coordinates", tuple("ABCD")),
        ("F/G canonicalization", ("F", "G")),
        ("D/H/I pitch interface", ("D", "H", "I")),
    ]
    palette = {
        "A": "#2864b7",
        "B": "#4484c7",
        "C": "#70a7d8",
        "D": "#18212f",
        "F": "#667085",
        "G": "#2864b7",
        "H": "#4f8f6f",
        "I": "#d97732",
    }
    width, panel_width, y_top, y_bottom = 1140, 380, 55, 265
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="300" viewBox="0 0 {width} 300">',
        STYLE,
    ]
    for panel_index, (title, arms) in enumerate(panels):
        x0 = panel_index * panel_width
        selected_long = [row for row in long_rows if row["arm"] in arms]
        selected_mean = [row for row in mean_rows if row["arm"] in arms]
        ys = [float(row["validation_bits_per_declared_fact"]) for row in selected_long]
        y_min, y_max = min(ys), max(ys)

        def project(
            epoch: float,
            value: float,
            panel_x: int = x0,
            lower: float = y_min,
            upper: float = y_max,
        ) -> tuple[float, float]:
            x = panel_x + 45 + (epoch - 100.0) * (panel_width - 65) / 1900.0
            y = y_bottom - (value - lower) * (y_bottom - y_top) / max(upper - lower, 1e-12)
            return x, y

        elements.extend(
            (
                f'<text x="{x0 + panel_width / 2}" y="24" text-anchor="middle" class="title">{html.escape(title)}</text>',
                f'<line x1="{x0 + 45}" y1="{y_bottom}" x2="{x0 + panel_width - 20}" y2="{y_bottom}" class="axis"/>',
                f'<line x1="{x0 + 45}" y1="{y_top}" x2="{x0 + 45}" y2="{y_bottom}" class="axis"/>',
                f'<text x="{x0 + panel_width / 2}" y="289" text-anchor="middle" class="axislabel">target-equivalent epochs</text>',
            )
        )
        grouped_long: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in selected_long:
            grouped_long[(row["arm"], row["seed"])].append(row)
        for (arm, _seed), rows in sorted(grouped_long.items()):
            points = " ".join(
                f"{x:.1f},{y:.1f}"
                for x, y in (
                    project(
                        float(row["target_equivalent_epochs"]),
                        float(row["validation_bits_per_declared_fact"]),
                    )
                    for row in rows
                )
            )
            elements.append(
                f'<polyline points="{points}" class="thin" stroke="{palette[arm]}"/>'
            )
            for row in rows:
                if row["selected_checkpoint"].lower() == "true":
                    x, y = project(
                        float(row["target_equivalent_epochs"]),
                        float(row["validation_bits_per_declared_fact"]),
                    )
                    elements.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="white" '
                        f'stroke="{palette[arm]}" stroke-width="1.5"/>'
                    )
        grouped_mean: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in selected_mean:
            grouped_mean[row["arm"]].append(row)
        for arm, rows in sorted(grouped_mean.items()):
            points = " ".join(
                f"{x:.1f},{y:.1f}"
                for x, y in (
                    project(
                        float(row["mean_target_equivalent_epochs"]),
                        float(row["mean_validation_bits_per_declared_fact"]),
                    )
                    for row in rows
                )
            )
            elements.append(
                f'<polyline points="{points}" class="mean" stroke="{palette[arm]}"/>'
            )
    elements.append("</svg>")
    output = output_root / "formal_learning_dynamics_reconstructed.svg"
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return output


def build_figures(artifact_root: Path, output_root: Path) -> dict[str, str]:
    ai: dict[str, Any] = json.loads(
        (artifact_root / "aggregate_results/pop1k7_ai_paper_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    by_arm = {
        row["arm"]: (
            float(row["mean_bits_per_declared_fact"]),
            float(row["sample_sd_bits_per_declared_fact"]),
        )
        for row in ai["arm_summary"]
    }
    svg = "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="280" viewBox="0 0 900 280">',
            STYLE,
            _panel(
                "(a) Coordinate gain: musical time",
                [(arm, *by_arm[arm]) for arm in "ABCD"],
                0,
            ),
            _panel(
                "(b) Factorization and relational excess",
                [(arm, *by_arm[arm]) for arm in ("D", "H", "E", "I")],
                300,
            ),
            _panel(
                "(c) Coordinate gain: canonicalization",
                [(arm, *by_arm[arm]) for arm in ("F", "G")],
                600,
            ),
            "</svg>",
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    figure = output_root / "figure2_reconstructed.svg"
    figure.write_text(svg + "\n", encoding="utf-8")

    context: dict[str, Any] = json.loads(
        (artifact_root / "context_probe/context_dependence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    context_values = [
        ("full", 100.0 * float(context["metrics"]["normal"]["combined_accuracy"]), 0.0),
        (
            "left replaced",
            100.0 * float(context["metrics"]["left_shuffled"]["combined_accuracy"]),
            0.0,
        ),
        (
            "right replaced",
            100.0 * float(context["metrics"]["right_shuffled"]["combined_accuracy"]),
            0.0,
        ),
        (
            "both replaced",
            100.0 * float(context["metrics"]["both_shuffled"]["combined_accuracy"]),
            0.0,
        ),
    ]
    context_svg = "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="280" viewBox="0 0 620 280">',
            STYLE,
            _panel(
                "Context-relation intervention on 6,112 held-out windows",
                context_values,
                0,
                620,
            ),
            "</svg>",
        )
    )
    context_figure = output_root / "context_relation_reconstructed.svg"
    context_figure.write_text(context_svg + "\n", encoding="utf-8")
    learning = _curve_figure(artifact_root, output_root)
    return {
        "figure2": str(figure),
        "context_relation": str(context_figure),
        "learning_dynamics": str(learning),
    }
