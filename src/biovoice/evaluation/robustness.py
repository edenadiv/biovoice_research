"""Robustness breakdown helpers by metadata buckets."""

from __future__ import annotations

import pandas as pd


def bucketed_accuracy(frame: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    """Compute decision accuracy per bucket."""
    rows = []
    for bucket, part in frame.groupby(bucket_column):
        rows.append(
            {
                "bucket": bucket,
                "count": len(part),
                "decision_accuracy": float((part["final_decision"] == part["label"]).mean()),
            }
        )
    return pd.DataFrame(rows)
