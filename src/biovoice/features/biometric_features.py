"""Enrollment-vs-probe mismatch features for explainable decision making."""

from __future__ import annotations

from typing import Mapping


def compare_feature_dicts(enrollment: Mapping[str, float], probe: Mapping[str, float]) -> dict[str, float]:
    """Create interpretable mismatch summaries between enrollment and probe."""
    result: dict[str, float] = {}
    keys = sorted(set(enrollment.keys()) & set(probe.keys()))
    for key in keys:
        delta = float(probe[key] - enrollment[key])
        result[f"{key}_delta"] = delta
        result[f"{key}_abs_delta"] = abs(delta)
    if keys:
        result["global_feature_abs_delta_mean"] = float(
            sum(result[f"{key}_abs_delta"] for key in keys) / len(keys)
        )
    return result
