"""Unit tests for audio preprocessing."""

from __future__ import annotations

import torch

from biovoice.data.preprocessing import preprocess_audio


def test_preprocessing_pads_to_target_duration() -> None:
    waveform = torch.randn(1, 8000)
    config = {
        "target_sample_rate": 16000,
        "mono": True,
        "loudness_normalize": True,
        "silence_trim": False,
        "pad_to_seconds": 1.0,
        "truncate_to_seconds": 1.0,
    }
    processed = preprocess_audio(waveform, 8000, config)
    assert processed.sample_rate == 16000
    assert processed.waveform.shape[-1] == 16000
