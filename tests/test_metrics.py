"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from biovoice.evaluation.calibration import expected_calibration_error
from biovoice.evaluation.metrics import classification_metrics, classwise_metrics_frame


def test_classification_metrics_include_accuracy() -> None:
    metrics = classification_metrics([0, 1, 1], [0, 1, 0], probabilities=[0.1, 0.9, 0.4])
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_expected_calibration_error_is_bounded() -> None:
    ece = expected_calibration_error(np.array([0, 1]), np.array([0.1, 0.9]), bins=5)
    assert 0.0 <= ece <= 1.0


def test_classwise_metrics_frame_reports_each_requested_label() -> None:
    frame = classwise_metrics_frame(
        ["target_bona_fide", "spoof", "wrong_speaker", "spoof"],
        ["target_bona_fide", "spoof", "target_bona_fide", "wrong_speaker"],
        ["target_bona_fide", "spoof", "wrong_speaker"],
    )
    assert list(frame["label"]) == ["target_bona_fide", "spoof", "wrong_speaker"]
    assert frame.loc[frame["label"] == "spoof", "support"].iloc[0] == 2
