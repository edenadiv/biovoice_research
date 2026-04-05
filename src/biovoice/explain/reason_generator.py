"""Human-readable explanation text generation."""

from __future__ import annotations

import pandas as pd


def generate_reasons(
    final_decision: str,
    sv_score: float,
    spoof_probability: float,
    top_features: list[dict[str, float | str]],
    suspicious_segments: pd.DataFrame,
) -> list[str]:
    """Generate short textual explanation cues for reports and notebooks."""
    reasons = [
        f"Final decision: {final_decision}. Speaker similarity was {sv_score:.3f} and spoof probability was {spoof_probability:.3f}.",
    ]
    if not suspicious_segments.empty:
        top_segment = suspicious_segments.iloc[0]
        reasons.append(
            "Most suspicious segment spans "
            f"{top_segment['start_seconds']:.2f}s to {top_segment['end_seconds']:.2f}s "
            f"with spoof probability {top_segment['spoof_probability']:.3f} "
            f"and speaker similarity {top_segment['speaker_similarity']:.3f}."
        )
    for item in top_features:
        reasons.append(f"Large enrollment-probe mismatch detected in {item['feature']}: {float(item['value']):.3f}.")
    return reasons
