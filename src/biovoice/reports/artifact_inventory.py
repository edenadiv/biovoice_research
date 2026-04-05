"""Artifact inventory helpers for supervisor-friendly run inspection.

The run directory intentionally contains many files. These helpers turn that
directory into a compact index so supervisors can find the important evidence
without manually browsing every subfolder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PLOT_INTERPRETATION_NOTES: dict[str, dict[str, str]] = {
    "class_balance.png": {
        "what": "Distribution of evaluation trials across the final decision classes.",
        "why": "Helps reviewers see whether later metrics may be biased by class imbalance.",
        "interpretation": "Large imbalance means accuracy alone may be misleading.",
    },
    "duration_histogram.png": {
        "what": "Distribution of probe durations after preprocessing.",
        "why": "Short clips often weaken both speaker verification and spoof detection.",
        "interpretation": "A narrow range means duration effects are limited; a wide range motivates robustness checks.",
    },
    "sv_loss_curves.png": {
        "what": "Train and validation loss for the speaker verification baseline.",
        "why": "Confirms the baseline actually trained and whether it overfit quickly.",
        "interpretation": "Diverging train/validation curves suggest overfitting or protocol mismatch.",
    },
    "spoof_loss_curves.png": {
        "what": "Train and validation loss for the spoof baseline.",
        "why": "Shows whether the anti-spoof branch learned anything stable on the available data.",
        "interpretation": "Flat curves can indicate an underpowered model or an uninformative dataset.",
    },
    "sv_roc.png": {
        "what": "ROC curve for target vs non-target verification.",
        "why": "Summarizes ranking quality across SV thresholds.",
        "interpretation": "Curves near the diagonal indicate poor separation between target and non-target trials.",
    },
    "spoof_roc.png": {
        "what": "ROC curve for bona fide vs spoof classification.",
        "why": "Shows how well spoof scores rank true spoofs above bona fide speech.",
        "interpretation": "Better curves move toward the upper-left corner.",
    },
    "sv_pr.png": {
        "what": "Precision-recall curve for the SV branch.",
        "why": "Complements ROC when positive and negative classes are uneven.",
        "interpretation": "Higher precision at useful recall values is better.",
    },
    "spoof_pr.png": {
        "what": "Precision-recall curve for the spoof branch.",
        "why": "Shows how positive spoof findings degrade as recall increases.",
        "interpretation": "Useful when the number of spoofs differs from bona fide trials.",
    },
    "sv_det.png": {
        "what": "DET curve for speaker verification.",
        "why": "Visualizes false-positive and false-negative trade-offs in verification terms.",
        "interpretation": "Curves closer to the origin indicate a better operating region.",
    },
    "spoof_det.png": {
        "what": "DET curve for spoof detection.",
        "why": "Shows the balance between missing spoofs and over-flagging bona fide speech.",
        "interpretation": "Useful for understanding threshold sensitivity beyond ROC-AUC.",
    },
    "target_vs_non_target_scores.png": {
        "what": "Histogram of SV scores for target and non-target trials.",
        "why": "Shows whether a single threshold can reasonably separate the two groups.",
        "interpretation": "Heavy overlap means the SV branch alone is not enough.",
    },
    "spoof_vs_bonafide_scores.png": {
        "what": "Histogram of spoof probabilities for bona fide and spoof probes.",
        "why": "Shows how much the spoof detector separates its two classes.",
        "interpretation": "Overlap indicates limited spoof-only reliability.",
    },
    "confusion_matrix.png": {
        "what": "Count confusion matrix for the final three-way decision.",
        "why": "Directly exposes which error type dominates in the fused system.",
        "interpretation": "Large off-diagonal counts show the practical failure mode.",
    },
    "normalized_confusion_matrix.png": {
        "what": "Row-normalized confusion matrix for the final three-way decision.",
        "why": "Makes classwise behavior easier to compare when class counts differ.",
        "interpretation": "Each row sums to one, so the diagonal is class recall.",
    },
    "reliability_diagram.png": {
        "what": "Calibration curve comparing predicted spoof probabilities with observed frequencies.",
        "why": "Probability quality matters for threshold interpretation and report trustworthiness.",
        "interpretation": "Curves close to the diagonal are better calibrated.",
    },
    "threshold_sweep_heatmap.png": {
        "what": "Decision accuracy over a grid of SV and spoof thresholds.",
        "why": "Shows whether the system is robust to reasonable threshold movement.",
        "interpretation": "Large flat high-value regions indicate a more stable operating point.",
    },
    "score_scatter.png": {
        "what": "Scatter plot of SV score versus spoof probability by class.",
        "why": "Useful for seeing whether classes occupy distinct regions of joint score space.",
        "interpretation": "Separation in this plane supports the case for late fusion.",
    },
    "ablation_summary.png": {
        "what": "Compact comparison of the four alpha experiment modes.",
        "why": "Shows whether fusion and interpretable features help relative to single-branch baselines.",
        "interpretation": "Treat differences cautiously on demo data; use them to inspect behavior, not to claim superiority.",
    },
    "waveform_with_suspicious_segments.png": {
        "what": "Probe waveform with highlighted suspicious regions.",
        "why": "Gives a direct visual anchor for segment-level explainability.",
        "interpretation": "Highlighted spans should be read alongside the spoof and similarity timelines.",
    },
    "spoof_score_over_time.png": {
        "what": "Segment-level spoof probability timeline.",
        "why": "Shows whether suspiciousness is concentrated in specific temporal regions.",
        "interpretation": "Sharp peaks may suggest localized artifacts rather than global corruption.",
    },
    "speaker_similarity_over_time.png": {
        "what": "Segment-level speaker similarity timeline.",
        "why": "Shows whether biometric consistency drifts across the probe.",
        "interpretation": "Drops in similarity can flag wrong-speaker or manipulated regions.",
    },
    "feature_contributions.png": {
        "what": "Bar chart of the largest interpretable feature mismatches between enrollment and probe audio.",
        "why": "Connects the final decision to concrete acoustic cues instead of only abstract model scores.",
        "interpretation": "Large positive bars indicate stronger mismatch evidence; treat them as supporting clues, not causal proof.",
    },
}


def _artifact_category(relative_path: Path) -> str:
    """Map a run-relative path to a human-readable artifact category."""
    if not relative_path.parts:
        return "root"
    return relative_path.parts[0]


def build_artifact_index(run_root: str | Path) -> pd.DataFrame:
    """Build a tabular inventory of files inside one run directory."""
    root = Path(run_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        rows.append(
            {
                "relative_path": str(relative),
                "category": _artifact_category(relative),
                "size_bytes": int(path.stat().st_size),
            }
        )
    return pd.DataFrame(rows)


def flatten_metric_dict(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested metric groups into a simple long-form table."""
    rows = []
    for group, values in metrics.items():
        for metric_name, value in values.items():
            rows.append({"group": group, "metric": metric_name, "value": value})
    return pd.DataFrame(rows)


def build_plot_inventory(run_root: str | Path) -> pd.DataFrame:
    """Summarize generated plots together with interpretation notes."""
    root = Path(run_root)
    rows: list[dict[str, str]] = []
    for path in sorted((root / "plots").glob("*.png")):
        note = PLOT_INTERPRETATION_NOTES.get(
            path.name,
            {
                "what": "Diagnostic plot.",
                "why": "Saved for inspection.",
                "interpretation": "Interpret together with the metrics and the experimental setup.",
            },
        )
        rows.append(
            {
                "filename": path.name,
                "what_it_shows": note["what"],
                "why_it_matters": note["why"],
                "interpretation_note": note["interpretation"],
            }
        )
    return pd.DataFrame(rows)
