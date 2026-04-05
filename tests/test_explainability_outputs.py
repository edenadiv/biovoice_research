"""Unit tests for explainability output helpers."""

from __future__ import annotations

import pandas as pd

from biovoice.explain.feature_attribution import top_feature_contributors
from biovoice.explain.segment_reasoning import rank_suspicious_segments


def test_top_feature_contributors_returns_ranked_rows() -> None:
    rows = top_feature_contributors({"f0_abs_delta": 0.2, "energy_abs_delta": 0.5}, top_k=1)
    assert rows[0]["feature"] == "energy_abs_delta"


def test_rank_suspicious_segments_adds_score() -> None:
    frame = pd.DataFrame(
        [
            {"start_seconds": 0.0, "end_seconds": 1.0, "spoof_probability": 0.2, "speaker_similarity": 0.9},
            {"start_seconds": 1.0, "end_seconds": 2.0, "spoof_probability": 0.8, "speaker_similarity": 0.2},
        ]
    )
    ranked = rank_suspicious_segments(frame, top_k=1)
    assert ranked.iloc[0]["spoof_probability"] == 0.8
