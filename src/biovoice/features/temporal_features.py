"""Temporal contour summaries used in explainable fusion."""

from __future__ import annotations

import numpy as np
import torch


def extract_temporal_features(waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
    """Compute simple temporal fluctuation summaries from the waveform."""
    signal = waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    frame = max(1, int(0.02 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    envelopes = []
    for start in range(0, max(len(signal) - frame + 1, 1), hop):
        chunk = signal[start : start + frame]
        if len(chunk) < frame:
            chunk = np.pad(chunk, (0, frame - len(chunk)))
        envelopes.append(np.sqrt(np.mean(chunk**2) + 1e-8))
    contour = np.asarray(envelopes, dtype=np.float32)
    diffs = np.diff(contour) if len(contour) > 1 else np.asarray([0.0], dtype=np.float32)
    return {
        "energy_delta_mean": float(np.mean(np.abs(diffs))),
        "energy_delta_std": float(np.std(diffs)),
        "contour_dynamic_range": float(np.max(contour) - np.min(contour)),
    }
