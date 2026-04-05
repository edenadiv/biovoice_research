"""Case-study assembly helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_case_analysis(
    trial_id: str,
    decision: str,
    reasons: list[str],
    segment_frame: pd.DataFrame,
    top_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a serializable case-study payload."""
    return {
        "trial_id": trial_id,
        "decision": decision,
        "reasons": reasons,
        "segment_analysis": segment_frame.to_dict(orient="records"),
        "top_features": top_features,
    }
