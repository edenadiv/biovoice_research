"""Spoof-detection specific metrics."""

from __future__ import annotations

import numpy as np

from biovoice.evaluation.metrics import classification_metrics
from biovoice.evaluation.sv_metrics import compute_eer


def spoof_metric_bundle(labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Binary spoof metric bundle including EER."""
    metrics = classification_metrics(labels, predictions, probabilities=probabilities)
    metrics["eer"] = compute_eer(labels, probabilities) if len(np.unique(labels)) == 2 else 0.0
    return metrics
