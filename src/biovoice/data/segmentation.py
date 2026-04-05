"""Segmentation helpers for segment-level SV and spoof analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(slots=True)
class SegmentInfo:
    """Metadata for one overlapping segment."""

    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize for report tables."""
        return asdict(self)


def segment_waveform(waveform: torch.Tensor, sample_rate: int, config: dict[str, float]) -> tuple[torch.Tensor, list[SegmentInfo]]:
    """Segment a waveform into overlapping windows.

    The returned tensor has shape ``[num_segments, channels, samples]``.
    """
    window = int(config["window_seconds"] * sample_rate)
    hop = int(config["hop_seconds"] * sample_rate)
    minimum = int(config.get("min_segment_seconds", config["window_seconds"]) * sample_rate)

    if waveform.shape[-1] < minimum:
        return waveform.unsqueeze(0), [
            SegmentInfo(
                index=0,
                start_sample=0,
                end_sample=waveform.shape[-1],
                start_seconds=0.0,
                end_seconds=waveform.shape[-1] / sample_rate,
                duration_seconds=waveform.shape[-1] / sample_rate,
            )
        ]

    segments: list[torch.Tensor] = []
    metadata: list[SegmentInfo] = []
    total = waveform.shape[-1]
    cursor = 0
    index = 0
    while cursor < total:
        end = min(cursor + window, total)
        chunk = waveform[..., cursor:end]
        if chunk.shape[-1] < minimum:
            break
        if chunk.shape[-1] < window:
            padding = window - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        segments.append(chunk)
        metadata.append(
            SegmentInfo(
                index=index,
                start_sample=cursor,
                end_sample=end,
                start_seconds=cursor / sample_rate,
                end_seconds=end / sample_rate,
                duration_seconds=(end - cursor) / sample_rate,
            )
        )
        cursor += hop
        index += 1
        if end == total:
            break
    return torch.stack(segments, dim=0), metadata
