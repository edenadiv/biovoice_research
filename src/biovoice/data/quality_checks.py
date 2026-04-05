"""Audio quality statistics and leakage-aware dataset checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(slots=True)
class AudioQualityStats:
    """Compact summary of an audio file used in preprocessing reports."""

    duration_seconds: float
    speech_ratio: float
    sample_rate: int
    clipping_ratio: float
    peak_amplitude: float
    rms: float
    snr_proxy_db: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass for tables and JSON artifacts."""
        return asdict(self)


def compute_speech_ratio(waveform: torch.Tensor, threshold: float = 0.02) -> float:
    """Estimate the fraction of frames with meaningful energy.

    This is intentionally simple and deterministic so it remains easy to audit.
    """
    if waveform.numel() == 0:
        return 0.0
    mono = waveform.mean(dim=0)
    frame_energy = mono.abs()
    return float((frame_energy > threshold).float().mean().item())


def compute_clipping_ratio(waveform: torch.Tensor, threshold: float = 0.99) -> float:
    """Estimate how much of the signal is close to clipping."""
    if waveform.numel() == 0:
        return 0.0
    return float((waveform.abs() >= threshold).float().mean().item())


def compute_snr_proxy_db(waveform: torch.Tensor) -> float:
    """A rough SNR proxy derived from percentile energy separation.

    It is not a replacement for a true SNR estimate, but it gives supervisors a
    practical quality indicator on both real and synthetic data.
    """
    if waveform.numel() == 0:
        return 0.0
    mono = waveform.mean(dim=0).abs().detach().cpu().numpy()
    high = np.percentile(mono, 95) + 1e-6
    low = np.percentile(mono, 20) + 1e-6
    return float(20.0 * np.log10(high / low))


def summarize_audio_quality(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.02) -> AudioQualityStats:
    """Compute the quality summary used throughout the repository."""
    if waveform.numel() == 0:
        return AudioQualityStats(
            duration_seconds=0.0,
            speech_ratio=0.0,
            sample_rate=sample_rate,
            clipping_ratio=0.0,
            peak_amplitude=0.0,
            rms=0.0,
            snr_proxy_db=0.0,
        )
    duration_seconds = waveform.shape[-1] / sample_rate
    peak = waveform.abs().max().item()
    rms = waveform.pow(2).mean().sqrt().item()
    return AudioQualityStats(
        duration_seconds=float(duration_seconds),
        speech_ratio=compute_speech_ratio(waveform, threshold=threshold),
        sample_rate=sample_rate,
        clipping_ratio=compute_clipping_ratio(waveform),
        peak_amplitude=float(peak),
        rms=float(rms),
        snr_proxy_db=compute_snr_proxy_db(waveform),
    )


def leakage_overlap_report(trials: pd.DataFrame) -> pd.DataFrame:
    """Flag obvious leakage patterns in a trial manifest."""
    rows: list[dict[str, Any]] = []
    for _, row in trials.iterrows():
        probe_path = row["probe_path"]
        enrollment_paths = set(row["enrollment_paths"])
        overlap = probe_path in enrollment_paths
        rows.append(
            {
                "trial_id": row["trial_id"],
                "speaker_id": row["speaker_id"],
                "probe_in_enrollment": overlap,
            }
        )
    return pd.DataFrame(rows)
