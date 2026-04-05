"""Configurable audio preprocessing for research experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from biovoice.data.quality_checks import AudioQualityStats, summarize_audio_quality
from biovoice.utils.audio_io import pad_or_truncate, resample_audio, rms_normalize, to_mono


@dataclass(slots=True)
class PreprocessedAudio:
    """Waveform plus quality metadata after the preprocessing pipeline."""

    waveform: torch.Tensor
    sample_rate: int
    stats: AudioQualityStats
    applied_steps: list[str]


def trim_silence(waveform: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Trim leading and trailing low-energy regions using an amplitude threshold."""
    mono = waveform.mean(dim=0)
    speech_mask = mono.abs() > threshold
    if not speech_mask.any():
        return waveform
    start = int(torch.argmax(speech_mask.float()).item())
    end = int(len(speech_mask) - torch.argmax(torch.flip(speech_mask, dims=[0]).float()).item())
    return waveform[..., start:end]


def preprocess_audio(waveform: torch.Tensor, sample_rate: int, config: dict[str, Any]) -> PreprocessedAudio:
    """Apply the repository's configurable preprocessing pipeline."""
    steps: list[str] = []
    target_rate = int(config["target_sample_rate"])
    processed = waveform.float()
    if config.get("mono", True):
        processed = to_mono(processed)
        steps.append("mono")
    if sample_rate != target_rate:
        processed = resample_audio(processed, sample_rate, target_rate)
        sample_rate = target_rate
        steps.append("resample")
    if config.get("loudness_normalize", False):
        processed = rms_normalize(processed)
        steps.append("rms_normalize")
    if config.get("silence_trim", False):
        processed = trim_silence(processed, threshold=float(config.get("silence_threshold", 0.02)))
        steps.append("trim_silence")
    if "pad_to_seconds" in config and config["pad_to_seconds"] is not None:
        target_samples = int(float(config["pad_to_seconds"]) * sample_rate)
        processed = pad_or_truncate(processed, target_samples)
        steps.append("pad_or_truncate")
    if "truncate_to_seconds" in config and config["truncate_to_seconds"] is not None:
        target_samples = int(float(config["truncate_to_seconds"]) * sample_rate)
        processed = pad_or_truncate(processed, target_samples)
        steps.append("truncate_cap")

    stats = summarize_audio_quality(processed, sample_rate, threshold=float(config.get("silence_threshold", 0.02)))
    return PreprocessedAudio(waveform=processed, sample_rate=sample_rate, stats=stats, applied_steps=steps)
