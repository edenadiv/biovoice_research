"""Segment-level inference helpers.

The repository keeps segment scoring lightweight by reusing the utterance-level
models over overlapping windows rather than introducing a second training stack.
"""

from __future__ import annotations

import torch

from biovoice.data.segmentation import segment_waveform


def score_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    segmentation_config: dict,
    speaker_encoder,
    enrollment_embedding: torch.Tensor,
    anti_spoof_model,
) -> tuple[list[dict], torch.Tensor]:
    """Score segments for spoof probability and speaker consistency."""
    segments, metadata = segment_waveform(waveform, sample_rate, segmentation_config)
    with torch.no_grad():
        segment_embeddings = speaker_encoder(segments)
        similarities = torch.nn.functional.cosine_similarity(
            segment_embeddings,
            enrollment_embedding.unsqueeze(0).expand_as(segment_embeddings),
            dim=-1,
        )
        spoof_output = anti_spoof_model(segments)
        spoof_probs = spoof_output["probability"]
    rows = []
    for info, sim, spoof in zip(metadata, similarities.tolist(), spoof_probs.tolist()):
        rows.append(
            {
                **info.to_dict(),
                "speaker_similarity": float(sim),
                "spoof_probability": float(spoof),
            }
        )
    return rows, segments
