"""Compact speaker encoder baseline for enrollment-conditioned verification."""

from __future__ import annotations

import torch
from torch import nn

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


class LogMelFrontend(nn.Module):
    """Shared log-Mel frontend for compact audio baselines."""

    def __init__(self, sample_rate: int, n_mels: int, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for the baseline frontends.")
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        mono = waveform.mean(dim=1)
        mel = self.mel(mono)
        return self.amplitude_to_db(mel).unsqueeze(1)


class SpeakerEncoder(nn.Module):
    """Compact convolutional embedding network."""

    def __init__(self, sample_rate: int, feature_config: dict, encoder_config: dict) -> None:
        super().__init__()
        hidden = int(encoder_config["hidden_channels"])
        embedding_dim = int(encoder_config["embedding_dim"])
        self.frontend = LogMelFrontend(sample_rate=sample_rate, **feature_config)
        self.backbone = nn.Sequential(
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
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(hidden * 2, embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.frontend(waveform)
        hidden = self.backbone(features)
        pooled = self.pool(hidden).flatten(1)
        embedding = self.projection(pooled)
        return torch.nn.functional.normalize(embedding, dim=-1)


class SpeakerClassificationModel(nn.Module):
    """Speaker encoder with a supervised speaker-ID head for baseline training."""

    def __init__(self, sample_rate: int, feature_config: dict, encoder_config: dict, num_speakers: int) -> None:
        super().__init__()
        self.encoder = SpeakerEncoder(sample_rate, feature_config, encoder_config)
        self.classifier = nn.Linear(int(encoder_config["embedding_dim"]), num_speakers)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encoder(waveform)
        logits = self.classifier(embedding)
        return {"embedding": embedding, "logits": logits}
