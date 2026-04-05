"""Unit tests for audio quality summary edge cases."""

from __future__ import annotations

import torch

from biovoice.data.quality_checks import summarize_audio_quality


def test_quality_summary_handles_empty_waveform() -> None:
    waveform = torch.zeros(1, 0)
    summary = summarize_audio_quality(waveform, sample_rate=16000)
    assert summary.duration_seconds == 0.0
    assert summary.speech_ratio == 0.0
    assert summary.peak_amplitude == 0.0
