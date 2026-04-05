"""Speaker verification metrics and threshold sweeps."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Approximate the equal-error rate from ROC points."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def target_non_target_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Summarize target and non-target score distributions."""
    target = scores[labels == 1]
    non_target = scores[labels == 0]
    return {
        "target_mean": float(target.mean()) if len(target) else 0.0,
        "target_std": float(target.std()) if len(target) else 0.0,
        "non_target_mean": float(non_target.mean()) if len(non_target) else 0.0,
        "non_target_std": float(non_target.std()) if len(non_target) else 0.0,
        "eer": compute_eer(labels, scores) if len(np.unique(labels)) == 2 else 0.0,
    }
