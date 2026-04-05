"""Unit tests for audio quality summary and leakage edge cases."""

from __future__ import annotations

import torch
import pandas as pd

from biovoice.data.quality_checks import assert_no_trial_leakage, summarize_audio_quality


def test_quality_summary_handles_empty_waveform() -> None:
    waveform = torch.zeros(1, 0)
    summary = summarize_audio_quality(waveform, sample_rate=16000)
    assert summary.duration_seconds == 0.0
    assert summary.speech_ratio == 0.0
    assert summary.peak_amplitude == 0.0


def test_trial_leakage_detection_catches_source_overlap() -> None:
    frame = pd.DataFrame(
        [
            {
                "trial_id": "trial_0001",
                "speaker_id": "speaker_a",
                "probe_path": "probe.wav",
                "enrollment_paths": ["enroll.wav"],
                "probe_source_recording_id": "src_001",
                "enrollment_source_recording_ids": "src_001|src_002",
            }
        ]
    )
    try:
        assert_no_trial_leakage(frame)
    except ValueError as error:
        assert "Leakage detected" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("Expected leakage validation to fail.")
