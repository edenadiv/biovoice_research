"""Calibration utilities for alpha evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Compute expected calibration error."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities >= start) & (probabilities < end if end < 1.0 else probabilities <= end)
        if not np.any(mask):
            continue
        accuracy = labels[mask].mean()
        confidence = probabilities[mask].mean()
        ece += np.abs(accuracy - confidence) * mask.mean()
    return float(ece)


def calibration_summary(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, object]:
    """Return calibration diagnostics ready for plotting and reporting."""
    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities, bins=bins),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
