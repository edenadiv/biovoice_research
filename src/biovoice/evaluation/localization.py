"""Segment localization metrics when labels are available."""

from __future__ import annotations

import numpy as np


def localization_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute simple precision/recall/F1 for suspicious segments."""
    tp = float(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = float(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = float(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1.0)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
