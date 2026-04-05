"""Unit tests for waveform segmentation."""

from __future__ import annotations

import torch

from biovoice.data.segmentation import segment_waveform


def test_segmentation_returns_overlapping_segments() -> None:
    waveform = torch.randn(1, 32000)
    segments, metadata = segment_waveform(
        waveform,
        sample_rate=16000,
        config={"window_seconds": 1.0, "hop_seconds": 0.5, "min_segment_seconds": 0.5},
    )
    assert segments.shape[0] >= 3
    assert metadata[0].start_seconds == 0.0
