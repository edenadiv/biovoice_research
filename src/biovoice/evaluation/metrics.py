"""Common metric helpers for binary and multiclass evaluation."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
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
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if len(set(y_true)) == 2 else "macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if probabilities is not None and len(set(y_true)) == 2:
        y_true_arr = np.asarray(y_true)
        prob_arr = np.asarray(probabilities)
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, prob_arr))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, prob_arr))
    return metrics
