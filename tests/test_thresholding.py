"""Unit tests for final-decision threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biovoice.evaluation.thresholding import (
    apply_thresholds,
    decision_metric_bundle,
    final_decision,
    select_best_thresholds,
    sweep_thresholds,
)


def test_thresholding_prefers_spoof_when_spoof_probability_is_high() -> None:
    decision = final_decision(0.8, 0.9, sv_threshold=0.5, spoof_threshold=0.5)
    assert decision == "spoof"


def test_thresholding_detects_wrong_speaker_when_similarity_is_low() -> None:
    decision = final_decision(0.2, 0.1, sv_threshold=0.5, spoof_threshold=0.5)
    assert decision == "wrong_speaker"


def test_threshold_sweep_reports_macro_f1_and_balanced_accuracy() -> None:
    frame = pd.DataFrame(
        {
            "sv_score": [0.9, 0.8, 0.2, 0.3],
            "spoof_probability": [0.1, 0.8, 0.2, 0.7],
            "label": ["target_bona_fide", "spoof", "wrong_speaker", "spoof"],
        }
    )
    sweep = sweep_thresholds(frame, np.asarray([0.5]), np.asarray([0.5]))
    assert {"decision_accuracy", "macro_f1", "balanced_accuracy"}.issubset(set(sweep.columns))


def test_select_best_thresholds_uses_requested_objective() -> None:
    sweep = pd.DataFrame(
        [
            {
                "sv_threshold": 0.4,
                "spoof_threshold": 0.4,
                "decision_accuracy": 0.7,
                "macro_f1": 0.5,
                "balanced_accuracy": 0.5,
                "weighted_f1": 0.6,
                "manual_review_rate": 0.0,
            },
            {
                "sv_threshold": 0.6,
                "spoof_threshold": 0.6,
                "decision_accuracy": 0.6,
                "macro_f1": 0.8,
                "balanced_accuracy": 0.7,
                "weighted_f1": 0.7,
                "manual_review_rate": 0.0,
            },
        ]
    )
    selected = select_best_thresholds(sweep, objective="macro_f1")
    assert selected["sv_threshold"] == 0.6
    assert selected["spoof_threshold"] == 0.6


def test_decision_metric_bundle_handles_manual_review_rows() -> None:
    frame = pd.DataFrame(
        {
            "label": ["spoof", "wrong_speaker", "target_bona_fide"],
            "final_decision": ["manual_review", "wrong_speaker", "target_bona_fide"],
        }
    )
    metrics = decision_metric_bundle(frame)
    assert abs(metrics["manual_review_rate"] - (1.0 / 3.0)) < 1e-9


def test_apply_thresholds_adds_requested_output_column() -> None:
    frame = pd.DataFrame({"sv_score": [0.8], "spoof_probability": [0.1]})
    decided = apply_thresholds(frame, 0.5, 0.5, output_column="tuned_decision")
    assert "tuned_decision" in decided.columns
