"""Segment-level suspiciousness scoring."""

from __future__ import annotations

import pandas as pd


def rank_suspicious_segments(segment_frame: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Combine spoof likelihood and speaker inconsistency into a suspiciousness score."""
    frame = segment_frame.copy()
    frame["suspiciousness"] = 0.6 * frame["spoof_probability"] + 0.4 * (1.0 - frame["speaker_similarity"])
    return frame.sort_values("suspiciousness", ascending=False).head(top_k).reset_index(drop=True)
