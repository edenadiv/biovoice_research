"""Late fusion and shallow trainable fusion head."""

from __future__ import annotations

import torch
from torch import nn


class LateFusionModel:
    """Rule-based late fusion for alpha baselines."""

    def __call__(self, sv_score: float, spoof_probability: float, feature_delta: float) -> float:
        """Combine normalized scores into a final target-bona-fide affinity score."""
        return float(0.55 * sv_score + 0.30 * (1.0 - spoof_probability) + 0.15 * (1.0 - min(feature_delta, 1.0)))


class TrainableFusionHead(nn.Module):
    """Shallow trainable head over branch scores and interpretable features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
