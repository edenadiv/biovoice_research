"""Enrollment aggregation utilities for claimed-speaker evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def aggregate_embeddings(embeddings: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate multiple enrollment embeddings into a single speaker template."""
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [num_enrollment, embedding_dim].")
    if method == "mean":
        return embeddings.mean(dim=0)
    if method == "median":
        return embeddings.median(dim=0).values
    raise ValueError(f"Unsupported aggregation method: {method}")


def summarize_enrollment(features: list[dict[str, float]]) -> dict[str, float]:
    """Summarize interpretable features across enrollment utterances."""
    if not features:
        return {}
    keys = sorted({key for item in features for key in item.keys()})
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([item[key] for item in features if key in item], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    return summary


def validate_trial_source_separation(trials: pd.DataFrame) -> pd.DataFrame:
    """Return a trial-level table that highlights source overlap risks."""
    flags: list[dict[str, Any]] = []
    for _, row in trials.iterrows():
        enrollment_roots = {PathLike(path).stem_root for path in row["enrollment_paths"]}
        probe_root = PathLike(row["probe_path"]).stem_root
        flags.append(
            {
                "trial_id": row["trial_id"],
                "speaker_id": row["speaker_id"],
                "source_overlap_risk": probe_root in enrollment_roots,
            }
        )
    return pd.DataFrame(flags)


class PathLike:
    """Tiny helper for source-recording heuristics based on filenames."""

    def __init__(self, path: str):
        self.path = path

    @property
    def stem_root(self) -> str:
        filename = self.path.split("/")[-1]
        return filename.split("_segment")[0].split(".")[0]
