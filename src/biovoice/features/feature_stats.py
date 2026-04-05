"""Feature aggregation helpers for tables and fusion inputs."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def feature_dicts_to_frame(records: Iterable[dict[str, float]]) -> pd.DataFrame:
    """Convert a stream of feature dictionaries into a DataFrame."""
    return pd.DataFrame(list(records))


def summarize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact summary statistics for feature reporting."""
    if frame.empty:
        return pd.DataFrame()
    summary = frame.describe().T.reset_index().rename(columns={"index": "feature"})
    return summary
