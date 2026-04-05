"""Gradient saliency for optional future qualitative analysis."""

from __future__ import annotations

import torch


def waveform_saliency(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Compute a simple input-gradient saliency map for a waveform."""
    model.eval()
    input_waveform = waveform.clone().detach().requires_grad_(True)
    output = model(input_waveform.unsqueeze(0))
    if "probability" in output:
        target = output["probability"].sum()
    else:
        target = output["logits"].max()
    target.backward()
    return input_waveform.grad.detach().abs()
