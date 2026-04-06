"""Evaluation plots for ROC, PR, DET, confusion matrices, and score distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, det_curve, precision_recall_curve, roc_curve

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_roc(labels: np.ndarray, scores: np.ndarray, path: str, title: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    """Plot a ROC curve with an explicit AUC label."""
    plt = prepare_figure(style)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    save_current_figure(path, dpi=dpi)


def plot_pr(labels: np.ndarray, scores: np.ndarray, path: str, title: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    """Plot a precision-recall curve for a binary task."""
    plt = prepare_figure(style)
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.plot(recall, precision, linewidth=2)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    save_current_figure(path, dpi=dpi)


def plot_det(labels: np.ndarray, scores: np.ndarray, path: str, title: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    """Plot a DET-style false-positive versus false-negative curve."""
    plt = prepare_figure(style)
    fpr, fnr, _ = det_curve(labels, scores)
    plt.plot(fpr, fnr, linewidth=2)
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    save_current_figure(path, dpi=dpi)


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    path: str,
    title: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
    normalize: bool = False,
) -> None:
    """Plot a count or row-normalized confusion matrix.

    Normalized views are easier for supervisors to interpret when the classes are
    imbalanced or when a demo set contains only a few examples of one decision.
    """
    plt = prepare_figure(style)
    values = matrix.values.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.where(row_sums == 0, 1.0, row_sums))
    image = plt.imshow(values, cmap="Blues", vmin=0.0)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell_value = values[i, j]
            text = f"{cell_value:.2f}" if normalize else str(int(matrix.iloc[i, j]))
            plt.text(j, i, text, ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_current_figure(path, dpi=dpi)


def plot_score_distributions(
    target_scores: np.ndarray,
    non_target_scores: np.ndarray,
    path: str,
    title: str,
    xlabel: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Overlay positive and negative score distributions on the same axes."""
    plt = prepare_figure(style)
    plt.hist(target_scores, bins=15, alpha=0.7, label="Positive", color="#2a9d8f")
    plt.hist(non_target_scores, bins=15, alpha=0.7, label="Negative", color="#e76f51")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    save_current_figure(path, dpi=dpi)


def plot_threshold_heatmap(
    frame: pd.DataFrame,
    path: str,
    title: str,
    value_column: str = "decision_accuracy",
    colorbar_label: str | None = None,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Plot a threshold sweep heatmap for one evaluation objective."""
    plt = prepare_figure(style)
    pivot = frame.pivot(index="spoof_threshold", columns="sv_threshold", values=value_column)
    image = plt.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(
        image,
        fraction=0.046,
        pad=0.04,
        label=colorbar_label or value_column.replace("_", " ").title(),
    )
    plt.xticks(range(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns], rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), [f"{value:.2f}" for value in pivot.index])
    plt.title(title)
    plt.xlabel("SV threshold")
    plt.ylabel("Spoof threshold")
    save_current_figure(path, dpi=dpi)


def plot_score_scatter(
    frame: pd.DataFrame,
    path: str,
    title: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Scatter SV score against spoof probability and color by final label."""
    plt = prepare_figure(style)
    palette = {
        "target_bona_fide": "#2a9d8f",
        "spoof": "#e76f51",
        "wrong_speaker": "#457b9d",
        "manual_review": "#6c757d",
    }
    for label, part in frame.groupby("label"):
        plt.scatter(
            part["sv_score"],
            part["spoof_probability"],
            label=label,
            s=45,
            alpha=0.85,
            color=palette.get(label, "#6c757d"),
            edgecolors="black",
            linewidths=0.3,
        )
    plt.title(title)
    plt.xlabel("Speaker similarity score")
    plt.ylabel("Spoof probability")
    plt.legend()
    save_current_figure(path, dpi=dpi)


def plot_score_by_class(
    frame: pd.DataFrame,
    score_column: str,
    path: str,
    title: str,
    ylabel: str,
    *,
    label_column: str = "label",
    labels: list[str] | None = None,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Plot one score distribution broken out by the true evaluation label.

    A compact boxplot is supervisor-friendly: it makes it obvious whether the
    score ranges for the three classes meaningfully overlap without requiring a
    more elaborate density estimator.
    """
    plt = prepare_figure(style)
    ordered_labels = labels or ["target_bona_fide", "spoof", "wrong_speaker"]
    series = [frame.loc[frame[label_column] == label, score_column].dropna().to_numpy() for label in ordered_labels]
    plt.boxplot(series, tick_labels=ordered_labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    save_current_figure(path, dpi=dpi)
