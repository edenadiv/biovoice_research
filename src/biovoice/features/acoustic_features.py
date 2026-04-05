"""Interpretable acoustic features for spoof-aware voice analysis.

Each feature is intentionally lightweight and auditable. The goal is not to
replace a learned representation, but to expose complementary signals that help
explain why enrollment and probe audio agree or diverge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.signal import find_peaks


FEATURE_METADATA: dict[str, dict[str, str]] = {
    "f0_mean": {
        "meaning": "Average estimated fundamental frequency across voiced frames.",
        "why_it_matters": "Deepfakes and wrong-speaker probes can shift the apparent pitch range away from enrollment speech.",
        "limitations": "Unvoiced speech and noisy conditions can make pitch estimation unstable.",
    },
    "f0_std": {
        "meaning": "Standard deviation of estimated F0 values.",
        "why_it_matters": "Flattened or overly erratic pitch contours can indicate synthesis artifacts or mismatched speakers.",
        "limitations": "Natural expressive speech can also have high pitch variance.",
    },
    "pitch_smoothness": {
        "meaning": "Inverse average frame-to-frame pitch jump magnitude.",
        "why_it_matters": "Synthetic speech can show unnatural pitch discontinuities or excessive smoothness.",
        "limitations": "Short clips can make contour summaries noisy.",
    },
    "energy_mean": {
        "meaning": "Average short-time energy across frames.",
        "why_it_matters": "Energy distribution helps characterize speaking style and synthesis artifacts.",
        "limitations": "Recording gain can alter absolute energy values.",
    },
    "energy_std": {
        "meaning": "Variation in short-time energy.",
        "why_it_matters": "Highly uniform energy may indicate oversmoothed synthesis, while very erratic energy may reflect noise.",
        "limitations": "Background noise can inflate the variance.",
    },
    "pause_ratio": {
        "meaning": "Fraction of low-energy frames interpreted as pauses or silence.",
        "why_it_matters": "Pause structure is part of speaking style and can diverge under synthetic generation.",
        "limitations": "Threshold-based pause detection is crude.",
    },
    "spectral_centroid_mean": {
        "meaning": "Average spectral centroid across frames.",
        "why_it_matters": "Spectral brightness differences can highlight channel mismatch or vocoder artifacts.",
        "limitations": "Microphone response can also shift this feature.",
    },
    "spectral_rolloff_mean": {
        "meaning": "Average 85 percent spectral rolloff.",
        "why_it_matters": "Helps summarize upper-frequency content and smoothing artifacts.",
        "limitations": "Sensitive to noise and bandwidth limitations.",
    },
    "spectral_tilt": {
        "meaning": "Low-vs-high frequency energy balance.",
        "why_it_matters": "Spoofed or re-synthesized audio may alter spectral tilt compared with enrollment speech.",
        "limitations": "Channel effects can confound interpretation.",
    },
    "zero_crossing_rate": {
        "meaning": "Average rate of sign changes in the waveform.",
        "why_it_matters": "Provides a simple proxy for noisiness and high-frequency content.",
        "limitations": "Not specific to spoofing and can vary with phonetic content.",
    },
}


def _frame_signal(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))
    frames = []
    for start in range(0, max(len(signal) - frame_size + 1, 1), hop_size):
        frame = signal[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        frames.append(frame)
    return np.stack(frames, axis=0)


def _estimate_pitch(frame: np.ndarray, sample_rate: int, f0_min: float = 60.0, f0_max: float = 350.0) -> float:
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    autocorrelation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    min_lag = int(sample_rate / f0_max)
    max_lag = int(sample_rate / f0_min)
    window = autocorrelation[min_lag:max_lag]
    if len(window) == 0:
        return 0.0
    peaks, _ = find_peaks(window)
    if len(peaks) == 0:
        lag = int(np.argmax(window)) + min_lag
    else:
        lag = peaks[np.argmax(window[peaks])] + min_lag
    return float(sample_rate / max(lag, 1))


def _spectral_features(frame: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sample_rate)
    magnitude_sum = spectrum.sum() + 1e-6
    centroid = float((freqs * spectrum).sum() / magnitude_sum)
    cumulative = np.cumsum(spectrum)
    rolloff_index = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(rolloff_index, len(freqs) - 1)])
    midpoint = len(spectrum) // 4
    low = spectrum[:midpoint].mean() + 1e-6
    high = spectrum[midpoint:].mean() + 1e-6
    tilt = float(np.log(low / high))
    return centroid, rolloff, tilt


def extract_acoustic_features(waveform: torch.Tensor, sample_rate: int) -> dict[str, float]:
    """Extract interpretable utterance-level acoustic features."""
    signal = waveform.mean(dim=0).detach().cpu().numpy().astype(np.float32)
    frame_size = int(0.03 * sample_rate)
    hop_size = int(0.01 * sample_rate)
    frames = _frame_signal(signal, frame_size=frame_size, hop_size=hop_size)
    energies = np.sqrt(np.mean(frames**2, axis=1) + 1e-8)
    pauses = energies < np.percentile(energies, 20)

    pitches = np.asarray([_estimate_pitch(frame, sample_rate) for frame in frames], dtype=np.float32)
    voiced = pitches > 0
    voiced_pitches = pitches[voiced] if voiced.any() else np.asarray([0.0], dtype=np.float32)
    pitch_diffs = np.abs(np.diff(voiced_pitches)) if len(voiced_pitches) > 1 else np.asarray([0.0], dtype=np.float32)

    spectral = np.asarray([_spectral_features(frame, sample_rate) for frame in frames], dtype=np.float32)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    return {
        "f0_mean": float(voiced_pitches.mean()),
        "f0_std": float(voiced_pitches.std()),
        "pitch_smoothness": float(1.0 / (pitch_diffs.mean() + 1e-4)),
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "pause_ratio": float(pauses.mean()),
        "spectral_centroid_mean": float(spectral[:, 0].mean()),
        "spectral_rolloff_mean": float(spectral[:, 1].mean()),
        "spectral_tilt": float(spectral[:, 2].mean()),
        "zero_crossing_rate": float(zcr.mean()),
    }


def explain_feature_set() -> dict[str, dict[str, str]]:
    """Return interpretation metadata for supervisor-facing reports."""
    return FEATURE_METADATA
