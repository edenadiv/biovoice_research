"""Compact anti-spoof baseline that scores utterances and segments."""

from __future__ import annotations

import torch
from torch import nn

from biovoice.models.speaker_encoder import LogMelFrontend


class AntiSpoofCNN(nn.Module):
    """Small spectrogram CNN for binary spoof detection."""

    def __init__(self, sample_rate: int, feature_config: dict, model_config: dict) -> None:
        super().__init__()
        hidden = int(model_config["hidden_channels"])
        self.frontend = LogMelFrontend(sample_rate=sample_rate, **feature_config)
        self.network = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.output = nn.Linear(hidden * 2, 1)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.frontend(waveform)
        hidden = self.network(features).flatten(1)
        logits = self.output(hidden).squeeze(-1)
        probability = torch.sigmoid(logits)
        return {"logits": logits, "probability": probability}
