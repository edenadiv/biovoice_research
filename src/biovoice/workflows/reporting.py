"""Reporting helpers that turn run artifacts into supervisor-friendly outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biovoice.reports.artifact_inventory import build_artifact_index
from biovoice.reports.run_report import build_run_report, save_run_report
from biovoice.reports.supervisor_report import build_supervisor_report, save_supervisor_report
from biovoice.reports.table_export import export_table
from biovoice.utils.serialization import save_json

from .common import ALPHA_LIMITATIONS, METRIC_INTERPRETATION_NOTES, build_alpha_checklist, save_inventory_tables


def _alpha_evidence_lines(dataset_review: dict[str, Any]) -> list[str]:
    """Build concise alpha evidence bullets tailored to the data source."""
    dataset_mode = str(dataset_review.get("dataset_mode", "demo"))
    if dataset_mode == "asvspoof2021_la":
        first_line = "ASVspoof 2019/2021 LA data staged and consumed end-to-end."
    elif dataset_mode == "real_private_corpus":
        first_line = "Private-corpus data staged and consumed end-to-end through canonical manifests."
    else:
        first_line = "Synthetic/demo data generated and consumed end-to-end."
    return [
        first_line,
        "Baseline SV run completed and saved history/checkpoint artifacts.",
        "Baseline spoof run completed and saved history/checkpoint artifacts.",
        "Fusion evaluation completed with saved predictions, metrics, and mandatory figures.",
        "Supervisor notebook can read saved artifacts from outputs.",
    ]


def _dataset_review_lines(dataset_review: dict[str, Any]) -> list[str]:
    """Format dataset review metadata as plain supervisor-facing bullets."""
    lines = [
        f"Dataset mode: {dataset_review['dataset_mode']}",
        f"Dataset name: {dataset_review['dataset_name']}",
        f"Split strategy: {dataset_review['split_strategy']}",
        f"Speaker-disjoint requirement: {dataset_review['require_speaker_disjoint']}",
        f"Speaker-disjoint status: {dataset_review.get('speaker_disjoint_status', 'unknown')}",
        f"Speaker-disjoint violations: {dataset_review['speaker_disjoint_violations']}",
        f"Leakage status: {dataset_review.get('trial_leakage_status', 'unknown')}",
        f"Trial leakage violations: {dataset_review['trial_leakage_violations']}",
        f"Trial counts by label: {dataset_review['trial_labels']}",
    ]
    if "quality_scan_mode" in dataset_review:
        lines.append(f"Quality scan mode: {dataset_review['quality_scan_mode']}")
    if "quality_measurement_counts" in dataset_review:
        lines.append(f"Quality measurement counts: {dataset_review['quality_measurement_counts']}")
    if "enrollment_policy_summary" in dataset_review:
        lines.append(f"Enrollment policy: {dataset_review['enrollment_policy_summary']}")
    if "wrong_speaker_policy_summary" in dataset_review:
        lines.append(f"Wrong-speaker policy: {dataset_review['wrong_speaker_policy_summary']}")
    return lines


def _real_data_interpretation_lines(dataset_review: dict[str, Any]) -> list[str]:
    """Explain how supervisors should interpret real-data baseline evidence."""
    dataset_mode = str(dataset_review.get("dataset_mode", "demo"))
    if dataset_mode == "demo":
        return [
            "This run proves the pipeline and reporting stack, not benchmark performance.",
            "Use demo numbers only to validate that the method can execute end-to-end.",
        ]
    return [
        "These metrics and figures are measured evidence for the current dataset and protocol only.",
        "This is still alpha-level evidence because the baselines are compact and robustness analysis remains limited.",
        "Do not overclaim real-world generalization, deployment readiness, or causal explainability from this run alone.",
    ]


def build_supervisor_summary(
    run_paths: Any,
    metrics: dict[str, dict[str, Any]],
    dataset_review: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the plain-language report payload consumed by the Markdown renderer."""
    dataset_mode = str(dataset_review.get("dataset_mode", "demo"))
    next_steps_tail = (
        "Treat the current numbers as measured ASVspoof baseline evidence and avoid broader real-world claims until robustness studies are complete."
        if dataset_mode == "asvspoof2021_la"
        else (
            "Treat the current numbers as measured private-corpus baseline evidence for this protocol only."
            if dataset_mode == "real_private_corpus"
            else "Treat the current numbers as alpha plumbing evidence because the run uses synthetic/demo data."
        )
    )
    return {
        "project_summary": "Enrollment-conditioned alpha baseline combining speaker verification, spoof detection, fusion logic, and interpretable mismatch analysis.",
        "alpha_evidence": _alpha_evidence_lines(dataset_review),
        "metrics": [
            f"SV EER: {metrics['sv'].get('eer', 0.0):.3f}",
            f"Spoof ROC-AUC: {metrics['spoof'].get('roc_auc', 0.0):.3f}",
            f"Joint Accuracy: {metrics['joint'].get('accuracy', 0.0):.3f}",
            f"Calibration ECE: {metrics['calibration']['ece']:.3f}",
        ],
        "dataset_review": _dataset_review_lines(dataset_review),
        "metric_interpretation": [
            METRIC_INTERPRETATION_NOTES["sv"],
            METRIC_INTERPRETATION_NOTES["spoof"],
            METRIC_INTERPRETATION_NOTES["joint"],
            METRIC_INTERPRETATION_NOTES["calibration"],
        ],
        "real_data_interpretation": _real_data_interpretation_lines(dataset_review),
        "figures": [str(path.relative_to(run_paths.root)) for path in sorted(run_paths.plots.glob("*.png"))],
        "artifact_map": [
            "Artifact index: reports/artifact_index.md",
            "Plot inventory: reports/plot_inventory.md",
            "Metric summary: reports/metric_summary.md",
            "Prediction table: tables/predictions.csv",
            f"Notebook-ready walkthrough artifacts live under: {run_paths.root}",
        ],
        "limitations": ALPHA_LIMITATIONS,
        "next_steps": [
            "Inspect the threshold heatmap to choose a more defensible operating region before claiming a decision policy.",
            "Use the plot inventory when presenting figures to supervisors so every figure is paired with an interpretation note.",
            next_steps_tail,
        ],
    }


def write_joint_run_outputs(
    run_paths: Any,
    metrics: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    dataset_review: dict[str, Any],
) -> None:
    """Persist metrics, tables, and supervisor-facing reports for a joint run."""
    save_json(metrics, run_paths.root / "metrics.json")
    export_table(predictions, run_paths.tables / "predictions.csv")
    comparison.to_csv(run_paths.tables / "ablation_summary.csv", index=False)
    save_json(dataset_review, run_paths.reports / "dataset_review.json")

    preliminary_artifact_frame = build_artifact_index(run_paths.root)
    save_run_report(
        build_run_report(
            "fusion_plus_interpretable_features",
            metrics,
            preliminary_artifact_frame["relative_path"].tolist()[:60],
            ALPHA_LIMITATIONS,
            alpha_checklist=None,
            interpretation_notes=METRIC_INTERPRETATION_NOTES,
        ),
        run_paths.reports / "run_report.md",
    )

    supervisor_summary = build_supervisor_summary(run_paths, metrics, dataset_review)
    save_supervisor_report(build_supervisor_report(supervisor_summary), run_paths.reports / "supervisor_report.md")

    save_inventory_tables(run_paths, metrics)
    alpha_checklist = build_alpha_checklist(run_paths)
    save_json(alpha_checklist, run_paths.reports / "alpha_exit_checklist.json")
    artifact_frame, _, _ = save_inventory_tables(run_paths, metrics)
    save_run_report(
        build_run_report(
            "fusion_plus_interpretable_features",
            metrics,
            artifact_frame["relative_path"].tolist()[:60],
            ALPHA_LIMITATIONS,
            alpha_checklist=alpha_checklist,
            interpretation_notes=METRIC_INTERPRETATION_NOTES,
        ),
        run_paths.reports / "run_report.md",
    )
    save_inventory_tables(run_paths, metrics)


def generate_supervisor_artifacts(config_path: str | Path) -> Path:
    """Ensure the end-to-end run exists and return its directory."""
    from .training import run_joint_workflow

    return run_joint_workflow(config_path)


def export_tables_workflow(config_path: str | Path) -> Path:
    """Trigger the end-to-end workflow so the expected tables exist."""
    from .training import run_joint_workflow

    return run_joint_workflow(config_path)
