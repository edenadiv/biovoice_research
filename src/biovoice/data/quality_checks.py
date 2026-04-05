"""Audio quality statistics and leakage-aware dataset checks.

The repository treats split safety as part of the scientific method rather than
an optional housekeeping detail. These helpers therefore support both
human-readable audit tables and hard validation failures when leakage is
detected.
"""

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
    """Flag obvious leakage patterns in a trial manifest.

    Two overlap modes are inspected:
    1. The exact probe path appears in the enrollment set.
    2. The probe shares a ``source_recording_id`` with one of the enrollment
       files, which can happen when a corpus stores multiple derivatives from
       the same source recording.
    """
    rows: list[dict[str, Any]] = []
    for _, row in trials.iterrows():
        probe_path = row["probe_path"]
        enrollment_paths = set(row["enrollment_paths"])
        overlap = probe_path in enrollment_paths
        probe_source = row.get("probe_source_recording_id")
        enrollment_sources = {
            item for item in str(row.get("enrollment_source_recording_ids", "")).split("|") if item
        }
        source_overlap = bool(probe_source) and probe_source in enrollment_sources
        rows.append(
            {
                "trial_id": row["trial_id"],
                "speaker_id": row["speaker_id"],
                "probe_in_enrollment": overlap,
                "source_recording_overlap": source_overlap,
                "has_leakage": overlap or source_overlap,
            }
        )
    return pd.DataFrame(rows)


def assert_no_trial_leakage(trials: pd.DataFrame) -> pd.DataFrame:
    """Validate that no trial leaks enrollment information into the probe.

    The returned frame is still useful for saved audit artifacts, but the
    function raises immediately when any row violates the trial-safe protocol.
    """
    report = leakage_overlap_report(trials)
    if report["has_leakage"].any():
        failing_trials = report.loc[report["has_leakage"], "trial_id"].tolist()
        raise ValueError(
            "Leakage detected between enrollment and probe audio for trials: "
            f"{failing_trials}"
        )
    return report


def speaker_split_report(utterances: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether a speaker appears in more than one split.

    For speaker-disjoint protocols, each speaker should appear in exactly one
    split. The report is also useful when speaker overlap is intentionally
    allowed, because it documents that choice explicitly for reviewers.
    """
    rows: list[dict[str, Any]] = []
    for speaker_id, frame in utterances.groupby("speaker_id"):
        splits = sorted(str(value) for value in frame["split"].dropna().unique())
        rows.append(
            {
                "speaker_id": speaker_id,
                "splits": "|".join(splits),
                "split_count": len(splits),
                "violates_speaker_disjoint": len(splits) > 1,
            }
        )
    return pd.DataFrame(rows)


def assert_speaker_disjoint(utterances: pd.DataFrame) -> pd.DataFrame:
    """Validate speaker-disjoint splits and return the audit report."""
    report = speaker_split_report(utterances)
    if report["violates_speaker_disjoint"].any():
        failing = report.loc[report["violates_speaker_disjoint"], "speaker_id"].tolist()
        raise ValueError(
            "Speaker-disjoint split validation failed for speakers appearing in "
            f"multiple splits: {failing}"
        )
    return report
