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
        first_line = "Leakage-safe ASVspoof 2019/2021 LA data staged and consumed end-to-end."
    elif dataset_mode == "real_private_corpus":
        first_line = "Leakage-safe private-corpus data staged and consumed end-to-end through canonical manifests."
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
        f"Evaluation trial counts by label: {dataset_review.get('test_trial_labels', dataset_review['trial_labels'])}",
    ]
    if dataset_review.get("validation_trial_labels"):
        lines.append(f"Validation trial counts by label: {dataset_review['validation_trial_labels']}")
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


def _baseline_comparison_lines(comparison: pd.DataFrame) -> list[str]:
    """Turn the baseline comparison table into concise supervisor bullets."""
    majority = comparison[comparison["mode"] == "majority_spoof"].iloc[0]
    sv_only = comparison[comparison["mode"] == "sv_only"].iloc[0]
    spoof_only = comparison[comparison["mode"] == "spoof_only"].iloc[0]
    tuned = comparison[comparison["mode"] == "fusion_tuned"].iloc[0]
    default = comparison[comparison["mode"] == "fusion_default"].iloc[0]
    return [
        (
            f"Majority baseline (`always spoof`) accuracy is {majority['accuracy']:.3f} "
            f"with macro F1 {majority['macro_f1']:.3f} and balanced accuracy {majority['balanced_accuracy']:.3f}."
        ),
        (
            f"SV-only reaches accuracy {sv_only['accuracy']:.3f}, macro F1 {sv_only['macro_f1']:.3f}, "
            f"and balanced accuracy {sv_only['balanced_accuracy']:.3f}; spoof-only reaches "
            f"accuracy {spoof_only['accuracy']:.3f}, macro F1 {spoof_only['macro_f1']:.3f}, "
            f"and balanced accuracy {spoof_only['balanced_accuracy']:.3f}."
        ),
        (
            f"Default fusion reaches accuracy {default['accuracy']:.3f}, macro F1 {default['macro_f1']:.3f}, "
            f"and balanced accuracy {default['balanced_accuracy']:.3f}; tuned fusion reaches "
            f"accuracy {tuned['accuracy']:.3f}, macro F1 {tuned['macro_f1']:.3f}, "
            f"and balanced accuracy {tuned['balanced_accuracy']:.3f}."
        ),
        (
            f"Tuned fusion changes relative to default thresholds: accuracy {tuned['accuracy'] - default['accuracy']:+.3f}, "
            f"macro F1 {tuned['macro_f1'] - default['macro_f1']:+.3f}, "
            f"balanced accuracy {tuned['balanced_accuracy'] - default['balanced_accuracy']:+.3f}."
        ),
        (
            "Because the class distribution is spoof-heavy, the majority baseline is a mandatory comparison and "
            "plain accuracy should not be interpreted alone."
        ),
    ]


def _threshold_selection_lines(metrics: dict[str, dict[str, Any]]) -> list[str]:
    """Explain how the active thresholds were chosen."""
    selected = metrics["threshold_selection"]
    return [
        f"Validation split used for tuning: {selected['search_split']}.",
        (
            f"Optimization objective: {selected['objective']} with selected SV threshold {selected['sv_threshold']:.3f} "
            f"and spoof threshold {selected['spoof_threshold']:.3f}."
        ),
        (
            f"Default thresholds were SV {selected['default_sv_threshold']:.3f} and spoof {selected['default_spoof_threshold']:.3f}; "
            f"`used_tuned_thresholds` was {selected['used_tuned_thresholds']}."
        ),
    ]


def _classwise_result_lines(classwise: pd.DataFrame) -> list[str]:
    """Summarize the classwise table as short readable bullets."""
    rows = classwise.sort_values("label")
    return [
        f"{row['label']}: precision {row['precision']:.3f}, recall {row['recall']:.3f}, F1 {row['f1']:.3f}, support {int(row['support'])}."
        for _, row in rows.iterrows()
    ]


def _decision_path_lines(decision_path: pd.DataFrame) -> list[str]:
    """Summarize how often the spoof gate dominates the final rule system."""
    if decision_path.empty:
        return ["Decision-path diagnostics were not available for this run."]
    rows = {row["path"]: row for _, row in decision_path.iterrows()}
    spoof_gate = rows.get("spoof_gate")
    sv_accept = rows.get("sv_accept_after_spoof_reject")
    sv_reject = rows.get("sv_reject_after_spoof_reject")
    lines: list[str] = []
    if spoof_gate is not None:
        lines.append(
            f"The spoof gate fired on {spoof_gate['fraction_of_trials']:.3f} of trials "
            f"({int(spoof_gate['count'])} / {int(decision_path['count'].sum())}), with a true-spoof fraction of "
            f"{spoof_gate['true_spoof_fraction']:.3f} inside that path."
        )
    if sv_accept is not None:
        lines.append(
            f"After the spoof gate stayed open, the SV accept path covered {sv_accept['fraction_of_trials']:.3f} of trials "
            f"and still contained a true-spoof fraction of {sv_accept['true_spoof_fraction']:.3f}, showing how many spoofs slip past branch B."
        )
    if sv_reject is not None:
        lines.append(
            f"The SV reject path after spoof rejection covered {sv_reject['fraction_of_trials']:.3f} of trials "
            f"with a true-spoof fraction of {sv_reject['true_spoof_fraction']:.3f}."
        )
    return lines


def _error_summary_lines(error_summary: pd.DataFrame) -> list[str]:
    """Highlight the largest confusion paths without overwhelming the reader."""
    if error_summary.empty:
        return ["No off-diagonal errors were recorded, which is unusual for a real-data baseline."]
    top_rows = error_summary.head(3)
    return [
        (
            f"{row['true_label']} -> {row['predicted_label']}: {int(row['count'])} trials "
            f"({row['rate_within_true_label']:.3f} of that true class)."
        )
        for _, row in top_rows.iterrows()
    ]


def build_supervisor_summary(
    run_paths: Any,
    metrics: dict[str, dict[str, Any]],
    dataset_review: dict[str, Any],
    comparison: pd.DataFrame,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the plain-language report payload consumed by the Markdown renderer."""
    dataset_mode = str(dataset_review.get("dataset_mode", "demo"))
    majority_accuracy = float(metrics["majority_baseline"].get("accuracy", 0.0))
    joint_accuracy = float(metrics["joint"].get("accuracy", 0.0))
    project_summary = (
        "Leakage-safe real-data alpha baseline combining speaker verification, spoof detection, "
        "validation-tuned fusion logic, and interpretable mismatch analysis."
        if dataset_mode != "demo"
        else "Enrollment-conditioned alpha baseline combining speaker verification, spoof detection, fusion logic, and interpretable mismatch analysis."
    )
    majority_warning = (
        "Current joint accuracy is below the trivial always-spoof baseline on this class distribution."
        if joint_accuracy < majority_accuracy
        else "Current joint accuracy is above the trivial always-spoof baseline, but the margin should still be interpreted cautiously."
    )
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
        "project_summary": project_summary,
        "alpha_evidence": _alpha_evidence_lines(dataset_review),
        "metrics": [
            f"Joint Macro F1: {metrics['joint'].get('macro_f1', 0.0):.3f}",
            f"Joint Balanced Accuracy: {metrics['joint'].get('balanced_accuracy', 0.0):.3f}",
            f"Joint Accuracy: {metrics['joint'].get('accuracy', 0.0):.3f}",
            f"Majority-Baseline Accuracy: {metrics['majority_baseline'].get('accuracy', 0.0):.3f}",
            f"SV EER: {metrics['sv'].get('eer', 0.0):.3f}",
            f"Spoof ROC-AUC: {metrics['spoof'].get('roc_auc', 0.0):.3f}",
            f"Calibration ECE: {metrics['calibration']['ece']:.3f}",
        ],
        "dataset_review": _dataset_review_lines(dataset_review),
        "metric_interpretation": [
            METRIC_INTERPRETATION_NOTES["sv"],
            METRIC_INTERPRETATION_NOTES["spoof"],
            METRIC_INTERPRETATION_NOTES["joint"],
            METRIC_INTERPRETATION_NOTES["calibration"],
            majority_warning,
        ],
        "baseline_comparison": _baseline_comparison_lines(comparison),
        "threshold_selection": _threshold_selection_lines(metrics),
        "classwise_results": _classwise_result_lines(analysis["classwise"]),
        "decision_path_summary": _decision_path_lines(analysis["decision_path"]),
        "error_summary": _error_summary_lines(analysis["error_summary"]),
        "real_data_interpretation": _real_data_interpretation_lines(dataset_review),
        "figures": [str(path.relative_to(run_paths.root)) for path in sorted(run_paths.plots.glob("*.png"))],
        "artifact_map": [
            "Artifact index: reports/artifact_index.md",
            "Plot inventory: reports/plot_inventory.md",
            "Metric summary: reports/metric_summary.md",
            "Prediction table: tables/predictions.csv",
            "Baseline comparison table: reports/baseline_comparison.md",
            "Classwise metrics table: reports/joint_classwise_metrics.md",
            "Decision-path summary table: reports/decision_path_summary.md",
            "Threshold comparison table: reports/threshold_comparison.md",
            f"Notebook-ready walkthrough artifacts live under: {run_paths.root}",
        ],
        "limitations": ALPHA_LIMITATIONS,
        "next_steps": [
            "Improve the spoof branch first, because it remains the main bottleneck in the final three-way decision.",
            "Use the threshold comparison and classwise metrics when discussing the run, not accuracy alone.",
            "Inspect the threshold heatmap and the per-class score plots before proposing any stronger decision rule.",
            next_steps_tail,
        ],
    }


def write_joint_run_outputs(
    run_paths: Any,
    metrics: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    dataset_review: dict[str, Any],
    *,
    analysis: dict[str, Any],
) -> None:
    """Persist metrics, tables, and supervisor-facing reports for a joint run."""
    save_json(metrics, run_paths.root / "metrics.json")
    predictions.to_csv(run_paths.root / "predictions.csv", index=False)
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
            baseline_comparison=_baseline_comparison_lines(comparison),
            threshold_selection=_threshold_selection_lines(metrics),
            classwise_results=_classwise_result_lines(analysis["classwise"]),
            decision_path_summary=_decision_path_lines(analysis["decision_path"]),
            error_summary=_error_summary_lines(analysis["error_summary"]),
        ),
        run_paths.reports / "run_report.md",
    )

    supervisor_summary = build_supervisor_summary(run_paths, metrics, dataset_review, comparison, analysis)
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
            baseline_comparison=_baseline_comparison_lines(comparison),
            threshold_selection=_threshold_selection_lines(metrics),
            classwise_results=_classwise_result_lines(analysis["classwise"]),
            decision_path_summary=_decision_path_lines(analysis["decision_path"]),
            error_summary=_error_summary_lines(analysis["error_summary"]),
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
