"""Lightweight waveform augmentations for research baselines."""

from __future__ import annotations

import torch


def add_noise(waveform: torch.Tensor, noise_scale: float = 0.01) -> torch.Tensor:
    """Inject Gaussian noise for simple robustness regularization."""
    return waveform + noise_scale * torch.randn_like(waveform)


def random_gain(waveform: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2) -> torch.Tensor:
    """Apply random gain scaling."""
    gain = torch.empty(1).uniform_(min_gain, max_gain).item()
    return waveform * gain
