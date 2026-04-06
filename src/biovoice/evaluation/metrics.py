"""Common metric helpers for binary and multiclass evaluation.

This module deliberately exposes a little more structure than a minimal
``accuracy`` helper because the project operates on a heavily imbalanced
three-way decision problem. A reviewer should be able to inspect both the
headline scalar metrics and the per-class behavior without recomputing
statistics by hand in a notebook.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute a compact but research-meaningful metric bundle."""
    unique_labels = set(y_true)
    binary_average = unique_labels.issubset({0, 1}) and len(unique_labels) == 2
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary" if binary_average else "macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary" if binary_average else "macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if binary_average else "macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if probabilities is not None and binary_average:
        y_true_arr = np.asarray(y_true)
        prob_arr = np.asarray(probabilities)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, prob_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, prob_arr))
    return metrics


def classwise_metrics_frame(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Return precision, recall, F1, and support for each named class.

    Parameters
    ----------
    y_true, y_pred:
        String-valued class labels for the final decision task.
    labels:
        Explicit class order to keep reports stable across runs.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    predicted = pd.Series(list(y_pred)).value_counts()
    rows = []
    for label, p, r, f, s in zip(labels, precision, recall, f1, support):
        rows.append(
            {
                "label": label,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
                "predicted_count": int(predicted.get(label, 0)),
            }
        )
    return pd.DataFrame(rows)
