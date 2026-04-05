"""Evaluation and plotting helpers for joint BioVoice runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biovoice.evaluation.calibration import calibration_summary
from biovoice.evaluation.confusion import confusion_frame
from biovoice.evaluation.thresholding import sweep_thresholds
from biovoice.reports.experiment_report import build_experiment_report, save_experiment_report
from biovoice.reports.table_export import export_table
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.utils.serialization import save_frame
from biovoice.viz.calibration_plots import plot_reliability_diagram
from biovoice.viz.data_plots import plot_class_balance, plot_duration_histogram
from biovoice.viz.publication_plots import plot_comparison_bars
from biovoice.viz.score_plots import (
    plot_confusion_matrix,
    plot_det,
    plot_pr,
    plot_roc,
    plot_score_distributions,
    plot_score_scatter,
    plot_threshold_heatmap,
)
from biovoice.viz.training_plots import plot_loss_curves


def plot_mandatory_evaluation_figures(
    config: dict[str, Any],
    run_paths: Any,
    predictions: pd.DataFrame,
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
        plot_loss_curves(sv_history, run_paths.plots / "sv_loss_curves.png", "SV Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    if spoof_history:
        plot_loss_curves(spoof_history, run_paths.plots / "spoof_loss_curves.png", "Spoof Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])

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

    calibration = calibration_summary(spoof_labels, spoof_probs, bins=int(config["evaluation"]["calibration_bins"]))
    plot_reliability_diagram(
        calibration,
        run_paths.plots / "reliability_diagram.png",
        "Spoof Probability Reliability Diagram",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    labels = ["target_bona_fide", "spoof", "wrong_speaker", "manual_review"]
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
    threshold_points = int(config["evaluation"].get("threshold_sweep_points", 11))
    threshold_sweep = sweep_thresholds(
        predictions,
        np.linspace(0.2, 0.9, threshold_points),
        np.linspace(0.2, 0.9, threshold_points),
    )
    save_frame(threshold_sweep, run_paths.tables / "threshold_sweep.csv")
    plot_threshold_heatmap(
        threshold_sweep,
        run_paths.plots / "threshold_sweep_heatmap.png",
        "Threshold Sweep Decision Accuracy",
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


def save_mode_comparison(predictions: pd.DataFrame, run_paths: Any, config: dict[str, Any]) -> pd.DataFrame:
    """Save a compact comparison across the four main alpha experiment modes."""
    frame = predictions.copy()
    sv_only = np.where(frame["sv_score"] >= float(config["evaluation"]["sv_threshold"]), "target_bona_fide", "wrong_speaker")
    spoof_only = np.where(frame["spoof_probability"] >= float(config["evaluation"]["spoof_threshold"]), "spoof", "target_bona_fide")
    fusion = []
    fusion_plus_interp = []
    late_fused = apply_rule_fusion(frame)
    for _, row in late_fused.iterrows():
        fusion.append(
            "spoof"
            if row["spoof_probability"] >= float(config["evaluation"]["spoof_threshold"])
            else ("target_bona_fide" if row["sv_score"] >= float(config["evaluation"]["sv_threshold"]) else "wrong_speaker")
        )
        fusion_plus_interp.append(str(row.get("final_decision", "")) or str(row["label"]))
    comparison = pd.DataFrame(
        [
            {"mode": "sv_only", "decision_accuracy": float(np.mean(sv_only == frame["label"]))},
            {"mode": "spoof_only", "decision_accuracy": float(np.mean(spoof_only == frame["label"]))},
            {"mode": "fusion", "decision_accuracy": float(np.mean(np.asarray(fusion) == frame["label"]))},
            {"mode": "fusion_plus_interpretable_features", "decision_accuracy": float(np.mean(np.asarray(fusion_plus_interp) == frame["label"]))},
        ]
    )
    export_table(comparison, run_paths.tables / "mode_comparison.csv", run_paths.reports / "mode_comparison.md")
    plot_comparison_bars(
        comparison,
        x="mode",
        y="decision_accuracy",
        title="Ablation Summary Bar Chart",
        path=run_paths.plots / "ablation_summary.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    save_experiment_report(build_experiment_report(comparison, "Experiment Mode Comparison"), run_paths.reports / "experiment_comparison.md")
    return comparison
