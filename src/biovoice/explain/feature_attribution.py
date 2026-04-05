"""Feature attribution utilities based on enrollment-vs-probe deltas."""

from __future__ import annotations


def top_feature_contributors(feature_deltas: dict[str, float], top_k: int = 5) -> list[dict[str, float | str]]:
    """Return the largest absolute interpretable feature deviations."""
    ranked = sorted(
        (
            {"feature": key, "value": value}
            for key, value in feature_deltas.items()
            if key.endswith("_abs_delta")
        ),
        key=lambda item: float(item["value"]),
        reverse=True,
    )
    return ranked[:top_k]
