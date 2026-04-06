"""Evaluation and analysis helpers for joint BioVoice runs.

This module is where the repository tries to be especially honest. The current
real-data baseline is useful as alpha evidence, but the three-way decision task
is heavily imbalanced, so raw accuracy alone is not a trustworthy headline.

The helpers below therefore make a few evaluation choices explicit:

- compare against a trivial always-``spoof`` baseline
- report classwise behavior, not just one scalar
- tune SV/spoof thresholds on validation data only
- save the threshold search itself as an inspectable artifact
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biovoice.evaluation.calibration import calibration_summary
from biovoice.evaluation.confusion import confusion_frame
from biovoice.evaluation.metrics import classwise_metrics_frame, classification_metrics
from biovoice.evaluation.spoof_metrics import spoof_metric_bundle
from biovoice.evaluation.sv_metrics import target_non_target_summary
from biovoice.evaluation.thresholding import (
    JOINT_LABELS,
    apply_thresholds,
    decision_metric_bundle,
    select_best_thresholds,
    sweep_thresholds,
)
from biovoice.reports.experiment_report import build_experiment_report, save_experiment_report
from biovoice.reports.table_export import export_table
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.utils.serialization import save_frame, save_json
from biovoice.viz.calibration_plots import plot_reliability_diagram
from biovoice.viz.data_plots import plot_class_balance, plot_duration_histogram
from biovoice.viz.publication_plots import plot_grouped_metric_bars
from biovoice.viz.score_plots import (
    plot_confusion_matrix,
    plot_det,
    plot_pr,
    plot_roc,
    plot_score_by_class,
    plot_score_distributions,
    plot_score_scatter,
    plot_threshold_heatmap,
)
from biovoice.viz.training_plots import plot_loss_curves


def _threshold_grid(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build the threshold grid from config with safe defaults."""
    evaluation_cfg = config["evaluation"]
    points = int(evaluation_cfg.get("threshold_sweep_points", 11))
    sv_thresholds = np.linspace(
        float(evaluation_cfg.get("sv_threshold_min", 0.2)),
        float(evaluation_cfg.get("sv_threshold_max", 0.9)),
        points,
    )
    spoof_thresholds = np.linspace(
        float(evaluation_cfg.get("spoof_threshold_min", 0.2)),
        float(evaluation_cfg.get("spoof_threshold_max", 0.9)),
        points,
    )
    return sv_thresholds, spoof_thresholds


def _binary_branch_metrics(
    predictions: pd.DataFrame,
    *,
    sv_threshold: float,
    spoof_threshold: float,
) -> dict[str, dict[str, float]]:
    """Compute SV and spoof branch metrics at the supplied operating point."""
    sv_binary_predictions = (predictions["sv_score"] >= sv_threshold).astype(int)
    spoof_binary_predictions = (predictions["spoof_probability"] >= spoof_threshold).astype(int)
    return {
        "sv": {
            **classification_metrics(
                predictions["speaker_match_label"],
                sv_binary_predictions,
                probabilities=predictions["sv_score"],
            ),
            **target_non_target_summary(
                predictions["sv_score"].to_numpy(),
                predictions["speaker_match_label"].to_numpy(),
            ),
        },
        "spoof": spoof_metric_bundle(
            predictions["spoof_label"].to_numpy(),
            predictions["spoof_probability"].to_numpy(),
            spoof_binary_predictions.to_numpy(),
        ),
    }


def _mode_decision_columns(
    frame: pd.DataFrame,
    *,
    default_sv_threshold: float,
    default_spoof_threshold: float,
    tuned_sv_threshold: float,
    tuned_spoof_threshold: float,
    manual_review_margin: float,
) -> pd.DataFrame:
    """Attach comparable baseline and tuned decision columns to one frame."""
    result = frame.copy()
    result["majority_spoof_decision"] = "spoof"
    result["sv_only_decision"] = np.where(
        result["sv_score"] >= default_sv_threshold,
        "target_bona_fide",
        "wrong_speaker",
    )
    result["spoof_only_decision"] = np.where(
        result["spoof_probability"] >= default_spoof_threshold,
        "spoof",
        "target_bona_fide",
    )
    result = apply_thresholds(
        result,
        default_sv_threshold,
        default_spoof_threshold,
        manual_review_margin=manual_review_margin,
        output_column="fusion_default_decision",
    )
    result = apply_thresholds(
        result,
        tuned_sv_threshold,
        tuned_spoof_threshold,
        manual_review_margin=manual_review_margin,
        output_column="fusion_tuned_decision",
    )
    result["final_decision"] = result["fusion_tuned_decision"]
    return result


def build_baseline_comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute side-by-side metrics for trivial and rule-based baselines."""
    rows: list[dict[str, Any]] = []
    mode_columns = {
        "majority_spoof": "majority_spoof_decision",
        "sv_only": "sv_only_decision",
        "spoof_only": "spoof_only_decision",
        "fusion_default": "fusion_default_decision",
        "fusion_tuned": "fusion_tuned_decision",
    }
    for mode, decision_column in mode_columns.items():
        metrics = decision_metric_bundle(frame, decision_column=decision_column)
        rows.append(
            {
                "mode": mode,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "manual_review_rate": float(metrics["manual_review_rate"]),
            }
        )
    comparison = pd.DataFrame(rows)
    majority_accuracy = float(comparison.loc[comparison["mode"] == "majority_spoof", "accuracy"].iloc[0])
    majority_macro_f1 = float(comparison.loc[comparison["mode"] == "majority_spoof", "macro_f1"].iloc[0])
    comparison["accuracy_minus_majority"] = comparison["accuracy"] - majority_accuracy
    comparison["macro_f1_minus_majority"] = comparison["macro_f1"] - majority_macro_f1
    return comparison


def build_error_summary_frame(frame: pd.DataFrame, *, decision_column: str = "final_decision") -> pd.DataFrame:
    """Summarize the dominant misclassification paths from the confusion matrix."""
    matrix = confusion_frame(frame["label"].tolist(), frame[decision_column].tolist(), labels=JOINT_LABELS)
    rows: list[dict[str, Any]] = []
    for true_label in JOINT_LABELS:
        row_total = int(matrix.loc[true_label].sum())
        for predicted_label in JOINT_LABELS:
            count = int(matrix.loc[true_label, predicted_label])
            if true_label == predicted_label or count == 0:
                continue
            rows.append(
                {
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "count": count,
                    "rate_within_true_label": float(count / max(row_total, 1)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count", "rate_within_true_label"])
    return pd.DataFrame(rows).sort_values(["count", "rate_within_true_label"], ascending=[False, False]).reset_index(drop=True)


def build_decision_path_summary_frame(
    frame: pd.DataFrame,
    *,
    sv_threshold: float,
    spoof_threshold: float,
) -> pd.DataFrame:
    """Count how often the spoof gate dominates the final three-way decision."""
    spoof_gate = frame["spoof_probability"] >= spoof_threshold
    sv_accept = (~spoof_gate) & (frame["sv_score"] >= sv_threshold)
    sv_reject = (~spoof_gate) & (frame["sv_score"] < sv_threshold)
    rows = [
        {
            "path": "spoof_gate",
            "count": int(spoof_gate.sum()),
            "fraction_of_trials": float(spoof_gate.mean()),
            "true_spoof_fraction": float(frame.loc[spoof_gate, "spoof_label"].mean()) if spoof_gate.any() else 0.0,
        },
        {
            "path": "sv_accept_after_spoof_reject",
            "count": int(sv_accept.sum()),
            "fraction_of_trials": float(sv_accept.mean()),
            "true_spoof_fraction": float(frame.loc[sv_accept, "spoof_label"].mean()) if sv_accept.any() else 0.0,
        },
        {
            "path": "sv_reject_after_spoof_reject",
            "count": int(sv_reject.sum()),
            "fraction_of_trials": float(sv_reject.mean()),
            "true_spoof_fraction": float(frame.loc[sv_reject, "spoof_label"].mean()) if sv_reject.any() else 0.0,
        },
    ]
    return pd.DataFrame(rows)


def _threshold_comparison_frame(
    frame: pd.DataFrame,
    *,
    default_sv_threshold: float,
    default_spoof_threshold: float,
    tuned_sv_threshold: float,
    tuned_spoof_threshold: float,
    manual_review_margin: float,
) -> pd.DataFrame:
    """Compare default and tuned threshold choices on the final evaluation split."""
    default_frame = apply_thresholds(
        frame,
        default_sv_threshold,
        default_spoof_threshold,
        manual_review_margin=manual_review_margin,
        output_column="decision",
    )
    tuned_frame = apply_thresholds(
        frame,
        tuned_sv_threshold,
        tuned_spoof_threshold,
        manual_review_margin=manual_review_margin,
        output_column="decision",
    )
    rows = []
    for name, candidate_frame, sv_threshold, spoof_threshold in [
        ("default", default_frame, default_sv_threshold, default_spoof_threshold),
        ("tuned", tuned_frame, tuned_sv_threshold, tuned_spoof_threshold),
    ]:
        metrics = decision_metric_bundle(candidate_frame, decision_column="decision")
        rows.append(
            {
                "threshold_set": name,
                "sv_threshold": float(sv_threshold),
                "spoof_threshold": float(spoof_threshold),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "manual_review_rate": float(metrics["manual_review_rate"]),
            }
        )
    return pd.DataFrame(rows)


def prepare_threshold_selection(
    config: dict[str, Any],
    validation_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute the validation-only threshold sweep and chosen operating point."""
    evaluation_cfg = config["evaluation"]
    threshold_objective = str(evaluation_cfg.get("threshold_objective", "macro_f1"))
    manual_review_margin = float(evaluation_cfg.get("manual_review_margin", 0.0))
    validation_predictions = apply_rule_fusion(validation_predictions)
    sv_thresholds, spoof_thresholds = _threshold_grid(config)
    threshold_sweep = sweep_thresholds(
        validation_predictions,
        sv_thresholds,
        spoof_thresholds,
        manual_review_margin=manual_review_margin,
    )
    selected_thresholds = select_best_thresholds(threshold_sweep, objective=threshold_objective)
    selected_thresholds["search_split"] = str(config["data"]["validation_split"])
    selected_thresholds["default_sv_threshold"] = float(evaluation_cfg["sv_threshold"])
    selected_thresholds["default_spoof_threshold"] = float(evaluation_cfg["spoof_threshold"])
    selected_thresholds["used_tuned_thresholds"] = bool(evaluation_cfg.get("use_tuned_thresholds", True))
    return threshold_sweep, selected_thresholds


def plot_mandatory_evaluation_figures(
    config: dict[str, Any],
    run_paths: Any,
    predictions: pd.DataFrame,
    *,
    threshold_sweep: pd.DataFrame,
    threshold_metric: str,
    sv_history: dict | None = None,
    spoof_history: dict | None = None,
) -> dict[str, Any]:
    """Generate the minimum alpha-review figure set plus core diagnostics."""
    plot_class_balance(
        predictions[["label"]].copy(),
        run_paths.plots / "class_balance.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    duration_frame = predictions.rename(columns={"probe_duration_seconds": "duration_seconds"})[["duration_seconds"]].copy()
    plot_duration_histogram(
        duration_frame,
        run_paths.plots / "duration_histogram.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    if sv_history:
        plot_loss_curves(
            sv_history,
            run_paths.plots / "sv_loss_curves.png",
            "SV Train/Validation Loss",
            dpi=config["plotting"]["dpi"],
            style=config["plotting"]["style"],
        )
    if spoof_history:
        plot_loss_curves(
            spoof_history,
            run_paths.plots / "spoof_loss_curves.png",
            "Spoof Train/Validation Loss",
            dpi=config["plotting"]["dpi"],
            style=config["plotting"]["style"],
        )

    sv_labels = predictions["speaker_match_label"].to_numpy(dtype=int)
    sv_scores = predictions["sv_score"].to_numpy(dtype=float)
    spoof_labels = predictions["spoof_label"].to_numpy(dtype=int)
    spoof_probs = predictions["spoof_probability"].to_numpy(dtype=float)
    if len(np.unique(sv_labels)) > 1:
        plot_roc(sv_labels, sv_scores, run_paths.plots / "sv_roc.png", "SV ROC Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_pr(sv_labels, sv_scores, run_paths.plots / "sv_pr.png", "SV Precision-Recall Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_det(sv_labels, sv_scores, run_paths.plots / "sv_det.png", "SV DET Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    if len(np.unique(spoof_labels)) > 1:
        plot_roc(spoof_labels, spoof_probs, run_paths.plots / "spoof_roc.png", "Spoof ROC Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_pr(spoof_labels, spoof_probs, run_paths.plots / "spoof_pr.png", "Spoof Precision-Recall Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_det(spoof_labels, spoof_probs, run_paths.plots / "spoof_det.png", "Spoof DET Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])

    target_scores = predictions.loc[predictions["speaker_match_label"] == 1, "sv_score"].to_numpy()
    non_target_scores = predictions.loc[predictions["speaker_match_label"] == 0, "sv_score"].to_numpy()
    plot_score_distributions(
        target_scores,
        non_target_scores,
        run_paths.plots / "target_vs_non_target_scores.png",
        "Target vs Non-Target Score Distributions",
        "SV Score",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    bona_scores = predictions.loc[predictions["spoof_label"] == 0, "spoof_probability"].to_numpy()
    spoof_scores = predictions.loc[predictions["spoof_label"] == 1, "spoof_probability"].to_numpy()
    plot_score_distributions(
        spoof_scores,
        bona_scores,
        run_paths.plots / "spoof_vs_bonafide_scores.png",
        "Spoof vs Bona Fide Score Distributions",
        "Spoof Probability",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_score_by_class(
        predictions,
        "sv_score",
        run_paths.plots / "sv_score_by_true_label.png",
        "SV Score by True Class",
        "SV Score",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_score_by_class(
        predictions,
        "spoof_probability",
        run_paths.plots / "spoof_probability_by_true_label.png",
        "Spoof Probability by True Class",
        "Spoof Probability",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    calibration = calibration_summary(spoof_labels, spoof_probs, bins=int(config["evaluation"]["calibration_bins"]))
    plot_reliability_diagram(
        calibration,
        run_paths.plots / "reliability_diagram.png",
        "Spoof Probability Reliability Diagram",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    labels = JOINT_LABELS + ["manual_review"]
    matrix = confusion_frame(predictions["label"].tolist(), predictions["final_decision"].tolist(), labels=labels)
    plot_confusion_matrix(matrix, run_paths.plots / "confusion_matrix.png", "Final Decision Confusion Matrix", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    plot_confusion_matrix(
        matrix,
        run_paths.plots / "normalized_confusion_matrix.png",
        "Normalized Final Decision Confusion Matrix",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
        normalize=True,
    )
    save_frame(threshold_sweep, run_paths.tables / "threshold_sweep.csv")
    plot_threshold_heatmap(
        threshold_sweep,
        run_paths.plots / "threshold_sweep_heatmap.png",
        f"Threshold Sweep ({threshold_metric.replace('_', ' ').title()})",
        value_column=threshold_metric,
        colorbar_label=threshold_metric.replace("_", " ").title(),
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_score_scatter(
        predictions,
        run_paths.plots / "score_scatter.png",
        "SV Score vs Spoof Probability",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    return calibration


def save_mode_comparison(comparison: pd.DataFrame, run_paths: Any, config: dict[str, Any]) -> pd.DataFrame:
    """Save the joint baseline comparison table and summary figure."""
    export_table(comparison, run_paths.tables / "baseline_comparison.csv", run_paths.reports / "baseline_comparison.md")
    plot_grouped_metric_bars(
        comparison,
        x="mode",
        metrics=["macro_f1", "balanced_accuracy", "accuracy"],
        title="Baseline Comparison Across Joint Decision Modes",
        path=run_paths.plots / "ablation_summary.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    save_experiment_report(
        build_experiment_report(comparison, "Baseline Comparison for the Final Decision Task"),
        run_paths.reports / "experiment_comparison.md",
    )
    return comparison


def evaluate_joint_predictions(
    config: dict[str, Any],
    run_paths: Any,
    predictions: pd.DataFrame,
    *,
    validation_predictions: pd.DataFrame,
    threshold_sweep: pd.DataFrame | None = None,
    selected_thresholds: dict[str, Any] | None = None,
    sv_history: dict | None = None,
    spoof_history: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Evaluate one joint run, save analysis tables, and return summary payloads."""
    evaluation_cfg = config["evaluation"]
    default_sv_threshold = float(evaluation_cfg["sv_threshold"])
    default_spoof_threshold = float(evaluation_cfg["spoof_threshold"])
    manual_review_margin = float(evaluation_cfg.get("manual_review_margin", 0.0))
    threshold_objective = str(evaluation_cfg.get("threshold_objective", "macro_f1"))
    use_tuned_thresholds = bool(evaluation_cfg.get("use_tuned_thresholds", True))

    if validation_predictions.empty:
        raise ValueError("Validation-based threshold tuning requires a non-empty validation prediction set.")

    predictions = apply_rule_fusion(predictions)
    if threshold_sweep is None or selected_thresholds is None:
        threshold_sweep, selected_thresholds = prepare_threshold_selection(config, validation_predictions)
    tuned_sv_threshold = float(selected_thresholds["sv_threshold"])
    tuned_spoof_threshold = float(selected_thresholds["spoof_threshold"])
    save_json(selected_thresholds, run_paths.reports / "threshold_selection.json")

    threshold_comparison = _threshold_comparison_frame(
        predictions,
        default_sv_threshold=default_sv_threshold,
        default_spoof_threshold=default_spoof_threshold,
        tuned_sv_threshold=tuned_sv_threshold,
        tuned_spoof_threshold=tuned_spoof_threshold,
        manual_review_margin=manual_review_margin,
    )
    export_table(
        threshold_comparison,
        run_paths.tables / "threshold_comparison.csv",
        run_paths.reports / "threshold_comparison.md",
    )

    predictions = _mode_decision_columns(
        predictions,
        default_sv_threshold=default_sv_threshold,
        default_spoof_threshold=default_spoof_threshold,
        tuned_sv_threshold=tuned_sv_threshold,
        tuned_spoof_threshold=tuned_spoof_threshold,
        manual_review_margin=manual_review_margin,
    )
    if not use_tuned_thresholds:
        predictions["final_decision"] = predictions["fusion_default_decision"]
    predictions["reason_1"] = predictions.apply(
        lambda row: (
            f"Final decision: {row['final_decision']}. "
            f"Speaker similarity was {float(row['sv_score']):.3f} and "
            f"spoof probability was {float(row['spoof_probability']):.3f}."
        ),
        axis=1,
    )

    active_sv_threshold = tuned_sv_threshold if use_tuned_thresholds else default_sv_threshold
    active_spoof_threshold = tuned_spoof_threshold if use_tuned_thresholds else default_spoof_threshold
    branch_metrics = _binary_branch_metrics(
        predictions,
        sv_threshold=active_sv_threshold,
        spoof_threshold=active_spoof_threshold,
    )
    joint_metrics = decision_metric_bundle(predictions)
    joint_default_metrics = decision_metric_bundle(predictions, decision_column="fusion_default_decision")
    sv_only_metrics = decision_metric_bundle(predictions, decision_column="sv_only_decision")
    spoof_only_metrics = decision_metric_bundle(predictions, decision_column="spoof_only_decision")
    majority_metrics = decision_metric_bundle(predictions, decision_column="majority_spoof_decision")
    calibration = plot_mandatory_evaluation_figures(
        config,
        run_paths,
        predictions,
        threshold_sweep=threshold_sweep,
        threshold_metric=threshold_objective,
        sv_history=sv_history,
        spoof_history=spoof_history,
    )

    classwise = classwise_metrics_frame(
        predictions["label"].tolist(),
        predictions["final_decision"].tolist(),
        JOINT_LABELS,
    )
    export_table(classwise, run_paths.tables / "joint_classwise_metrics.csv", run_paths.reports / "joint_classwise_metrics.md")

    error_summary = build_error_summary_frame(predictions)
    export_table(error_summary, run_paths.tables / "error_summary.csv", run_paths.reports / "error_summary.md")

    decision_path = build_decision_path_summary_frame(
        predictions,
        sv_threshold=active_sv_threshold,
        spoof_threshold=active_spoof_threshold,
    )
    export_table(decision_path, run_paths.tables / "decision_path_summary.csv", run_paths.reports / "decision_path_summary.md")

    comparison = build_baseline_comparison_frame(predictions)
    comparison = save_mode_comparison(comparison, run_paths, config)

    metrics = {
        **branch_metrics,
        "joint": joint_metrics,
        "joint_default": joint_default_metrics,
        "joint_sv_only": sv_only_metrics,
        "joint_spoof_only": spoof_only_metrics,
        "majority_baseline": majority_metrics,
        "calibration": {"brier_score": calibration["brier_score"], "ece": calibration["ece"]},
        "threshold_selection": selected_thresholds,
    }

    analysis = {
        "classwise": classwise,
        "error_summary": error_summary,
        "decision_path": decision_path,
        "threshold_comparison": threshold_comparison,
        "selected_thresholds": selected_thresholds,
        "active_sv_threshold": active_sv_threshold,
        "active_spoof_threshold": active_spoof_threshold,
    }
    return predictions, metrics, comparison, analysis
