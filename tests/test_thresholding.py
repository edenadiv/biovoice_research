"""Unit tests for final-decision threshold logic."""

from __future__ import annotations

from biovoice.evaluation.thresholding import final_decision


def test_thresholding_prefers_spoof_when_spoof_probability_is_high() -> None:
    decision = final_decision(0.8, 0.9, sv_threshold=0.5, spoof_threshold=0.5)
    assert decision == "spoof"


def test_thresholding_detects_wrong_speaker_when_similarity_is_low() -> None:
    decision = final_decision(0.2, 0.1, sv_threshold=0.5, spoof_threshold=0.5)
    assert decision == "wrong_speaker"
