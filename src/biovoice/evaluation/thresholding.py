"""Decision logic and threshold sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def final_decision(
    sv_score: float,
    spoof_probability: float,
    sv_threshold: float,
    spoof_threshold: float,
    manual_review_margin: float = 0.0,
) -> str:
    """Map branch scores to the final alpha decision labels."""
    near_sv = abs(sv_score - sv_threshold) <= manual_review_margin
    near_spoof = abs(spoof_probability - spoof_threshold) <= manual_review_margin
    if near_sv or near_spoof:
        return "manual_review"
    if spoof_probability >= spoof_threshold:
        return "spoof"
    if sv_score >= sv_threshold:
        return "target_bona_fide"
    return "wrong_speaker"


def sweep_thresholds(frame: pd.DataFrame, sv_thresholds: np.ndarray, spoof_thresholds: np.ndarray) -> pd.DataFrame:
    """Evaluate a grid of decision thresholds."""
    rows = []
    for sv_threshold in sv_thresholds:
        for spoof_threshold in spoof_thresholds:
            decisions = [
                final_decision(sv, spoof, sv_threshold, spoof_threshold)
                for sv, spoof in zip(frame["sv_score"], frame["spoof_probability"])
            ]
            accuracy = float(np.mean(np.asarray(decisions) == frame["label"].to_numpy()))
            rows.append(
                {
                    "sv_threshold": float(sv_threshold),
                    "spoof_threshold": float(spoof_threshold),
                    "decision_accuracy": accuracy,
                }
            )
    return pd.DataFrame(rows)
