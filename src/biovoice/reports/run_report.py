"""Per-run Markdown summary generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_run_report(
    experiment_name: str,
    metrics: dict[str, Any],
    artifact_index: list[str],
    limitations: list[str],
    alpha_checklist: dict[str, bool] | None = None,
    interpretation_notes: dict[str, str] | None = None,
    baseline_comparison: list[str] | None = None,
    threshold_selection: list[str] | None = None,
    classwise_results: list[str] | None = None,
    decision_path_summary: list[str] | None = None,
    error_summary: list[str] | None = None,
) -> str:
    """Render a readable run report for researchers and supervisors."""
    lines = [
        f"# Run Report: {experiment_name}",
        "",
        "## Purpose",
        "This report summarizes measured outputs from one experiment run. It is intended to help a reviewer understand what was executed, what artifacts were produced, and how cautiously to interpret the results.",
        "",
    ]
    if alpha_checklist:
        lines.extend(["## Alpha Exit Checklist"])
        for item, status in alpha_checklist.items():
            marker = "PASS" if status else "FAIL"
            lines.append(f"- {marker}: {item}")
        lines.append("")
    lines.extend(
        [
        "## Metrics",
        ]
    )
    for group, values in metrics.items():
        lines.append(f"### {group}")
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if interpretation_notes:
        lines.append("## Interpretation Notes")
        for key, note in interpretation_notes.items():
            lines.append(f"- {key}: {note}")
        lines.append("")
    if baseline_comparison:
        lines.append("## Baseline Comparison")
        for item in baseline_comparison:
            lines.append(f"- {item}")
        lines.append("")
    if threshold_selection:
        lines.append("## Threshold Selection")
        for item in threshold_selection:
            lines.append(f"- {item}")
        lines.append("")
    if classwise_results:
        lines.append("## Classwise Results")
        for item in classwise_results:
            lines.append(f"- {item}")
        lines.append("")
    if decision_path_summary:
        lines.append("## Decision Path Summary")
        for item in decision_path_summary:
            lines.append(f"- {item}")
        lines.append("")
    if error_summary:
        lines.append("## Error Summary")
        for item in error_summary:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Artifacts")
    for item in artifact_index:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Limitations")
    for item in limitations:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Integrity Statement")
    lines.append("- This report contains measured outputs from the current run only.")
    lines.append("- No fabricated results or unsupported claims are included.")
    return "\n".join(lines)


def save_run_report(markdown: str, path: str | Path) -> None:
    """Persist the run report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
