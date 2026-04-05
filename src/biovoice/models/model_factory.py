"""Factory functions for building baseline models from config."""

from __future__ import annotations

from biovoice.models.anti_spoof_model import AntiSpoofCNN
from biovoice.models.fusion_model import TrainableFusionHead
from biovoice.models.speaker_encoder import SpeakerClassificationModel, SpeakerEncoder


def build_speaker_model(config: dict, num_speakers: int) -> SpeakerClassificationModel:
    """Build the supervised speaker classification baseline."""
    return SpeakerClassificationModel(
        sample_rate=int(config["data"]["sample_rate"]),
        feature_config=config["model"]["feature"],
        encoder_config=config["model"]["speaker_encoder"],
        num_speakers=num_speakers,
    )


def build_speaker_encoder(config: dict) -> SpeakerEncoder:
    """Build a standalone speaker encoder for inference."""
    return SpeakerEncoder(
        sample_rate=int(config["data"]["sample_rate"]),
        feature_config=config["model"]["feature"],
        encoder_config=config["model"]["speaker_encoder"],
    )


def build_anti_spoof_model(config: dict) -> AntiSpoofCNN:
    """Build the baseline anti-spoof model."""
    return AntiSpoofCNN(
        sample_rate=int(config["data"]["sample_rate"]),
        feature_config=config["model"]["feature"],
        model_config=config["model"]["anti_spoof"],
    )


def build_fusion_head(input_dim: int) -> TrainableFusionHead:
    """Build the shallow fusion head."""
    return TrainableFusionHead(input_dim=input_dim)
