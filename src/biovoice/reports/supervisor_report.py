"""Supervisor-facing report assembly."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_supervisor_report(summary: dict[str, Any]) -> str:
    """Render a supervisor-oriented Markdown summary.

    The output deliberately uses plain language first so a supervisor can scan
    it quickly before diving into the raw metrics and figures.
    """
    lines = [
        "# Supervisor Review Report",
        "",
        "## Project Summary",
        summary["project_summary"],
        "",
        "## How To Read This Report",
        "Start with the alpha-exit evidence, then inspect the key metrics, then open the mandatory figures and the walkthrough notebook. Treat the current run numbers as measured baseline evidence rather than broad real-world performance claims.",
        "",
        "## Alpha Exit Criteria Evidence",
    ]
    for item in summary["alpha_evidence"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Key Metrics",
        ]
    )
    for item in summary["metrics"]:
        lines.append(f"- {item}")
    if "dataset_review" in summary:
        lines.extend(["", "## Dataset Review"])
        for item in summary["dataset_review"]:
            lines.append(f"- {item}")
    if "metric_interpretation" in summary:
        lines.extend(["", "## Metric Interpretation"])
        for item in summary["metric_interpretation"]:
            lines.append(f"- {item}")
    if "baseline_comparison" in summary:
        lines.extend(["", "## Baseline Comparison"])
        for item in summary["baseline_comparison"]:
            lines.append(f"- {item}")
    if "threshold_selection" in summary:
        lines.extend(["", "## Threshold Selection"])
        for item in summary["threshold_selection"]:
            lines.append(f"- {item}")
    if "classwise_results" in summary:
        lines.extend(["", "## Classwise Results"])
        for item in summary["classwise_results"]:
            lines.append(f"- {item}")
    if "decision_path_summary" in summary:
        lines.extend(["", "## Decision Path Summary"])
        for item in summary["decision_path_summary"]:
            lines.append(f"- {item}")
    if "error_summary" in summary:
        lines.extend(["", "## Error Summary"])
        for item in summary["error_summary"]:
            lines.append(f"- {item}")
    if "real_data_interpretation" in summary:
        lines.extend(["", "## How To Interpret This Real-Data Baseline"])
        for item in summary["real_data_interpretation"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Mandatory Figures",
        ]
    )
    for item in summary["figures"]:
        lines.append(f"- {item}")
    if "artifact_map" in summary:
        lines.extend(["", "## Artifact Map"])
        for item in summary["artifact_map"]:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Limitations",
        ]
    )
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    if "next_steps" in summary:
        lines.extend(["", "## Recommended Next Steps"])
        for item in summary["next_steps"]:
            lines.append(f"- {item}")
    return "\n".join(lines)


def save_supervisor_report(markdown: str, path: str | Path) -> None:
    """Persist the supervisor report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
