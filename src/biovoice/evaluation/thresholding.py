"""Decision logic, threshold sweeps, and threshold selection.

The alpha baseline uses a simple two-threshold decision rule:

- high spoof probability overrides the final decision to ``spoof``
- otherwise the SV score separates ``target_bona_fide`` from ``wrong_speaker``

That rule is intentionally simple, but the threshold choice materially affects
reported performance on the imbalanced three-way task. The helpers in this
module keep threshold search explicit, validation-driven, and auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biovoice.evaluation.metrics import classification_metrics


JOINT_LABELS = ["wrong_speaker", "spoof", "target_bona_fide"]
OBJECTIVE_COLUMNS = {"accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"}


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


def apply_thresholds(
    frame: pd.DataFrame,
    sv_threshold: float,
    spoof_threshold: float,
    manual_review_margin: float = 0.0,
    *,
    output_column: str = "final_decision",
) -> pd.DataFrame:
    """Apply one threshold pair to a score frame and return a copy.

    The input frame must contain ``sv_score`` and ``spoof_probability``.
    """
    result = frame.copy()
    result[output_column] = [
        final_decision(sv, spoof, sv_threshold, spoof_threshold, manual_review_margin=manual_review_margin)
        for sv, spoof in zip(result["sv_score"], result["spoof_probability"])
    ]
    return result


def decision_metric_bundle(
    frame: pd.DataFrame,
    *,
    decision_column: str = "final_decision",
    label_column: str = "label",
) -> dict[str, float]:
    """Compute multiclass metrics for one decision column.

    ``manual_review`` rows are excluded from the classification metrics and
    tracked separately via ``manual_review_rate`` so reviewers can see whether a
    threshold setting is only "good" because it abstains too often.
    """
    valid = frame[decision_column] != "manual_review"
    if not valid.any():
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "manual_review_rate": 1.0,
        }
    mapping = {label: index for index, label in enumerate(JOINT_LABELS)}
    true = frame.loc[valid, label_column].map(mapping).to_numpy()
    pred = frame.loc[valid, decision_column].map(mapping).to_numpy()
    metrics = classification_metrics(true, pred)
    metrics["manual_review_rate"] = 1.0 - float(valid.mean())
    return metrics


def sweep_thresholds(
    frame: pd.DataFrame,
    sv_thresholds: np.ndarray,
    spoof_thresholds: np.ndarray,
    *,
    manual_review_margin: float = 0.0,
) -> pd.DataFrame:
    """Evaluate a grid of decision thresholds using several objectives."""
    rows = []
    for sv_threshold in sv_thresholds:
        for spoof_threshold in spoof_thresholds:
            decided = apply_thresholds(
                frame,
                float(sv_threshold),
                float(spoof_threshold),
                manual_review_margin=manual_review_margin,
            )
            metrics = decision_metric_bundle(decided)
            rows.append(
                {
                    "sv_threshold": float(sv_threshold),
                    "spoof_threshold": float(spoof_threshold),
                    "decision_accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(metrics["macro_f1"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "manual_review_rate": float(metrics["manual_review_rate"]),
                }
            )
    return pd.DataFrame(rows)


def select_best_thresholds(
    threshold_sweep: pd.DataFrame,
    *,
    objective: str = "macro_f1",
) -> dict[str, float]:
    """Pick the best threshold pair from a saved sweep table.

    Ties are broken conservatively:
    1. lower manual-review rate
    2. higher balanced accuracy
    3. higher plain accuracy
    4. lower spoof threshold then lower SV threshold for deterministic output
    """
    if objective not in OBJECTIVE_COLUMNS:
        raise ValueError(
            f"Unsupported threshold objective '{objective}'. "
            f"Expected one of {sorted(OBJECTIVE_COLUMNS)}."
        )
    ranked = threshold_sweep.sort_values(
        by=[objective, "manual_review_rate", "balanced_accuracy", "decision_accuracy", "spoof_threshold", "sv_threshold"],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)
    best = ranked.iloc[0]
    return {
        "objective": objective,
        "sv_threshold": float(best["sv_threshold"]),
        "spoof_threshold": float(best["spoof_threshold"]),
        "objective_value": float(best[objective]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "decision_accuracy": float(best["decision_accuracy"]),
        "manual_review_rate": float(best["manual_review_rate"]),
    }
