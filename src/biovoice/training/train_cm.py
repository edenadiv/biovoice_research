"""Spoof-detection baseline training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from biovoice.data.loading import UtteranceDataset, collate_utterances
from biovoice.data.manifests import load_manifest
from biovoice.models.losses import spoof_classification_loss
from biovoice.models.model_factory import build_anti_spoof_model
from biovoice.training.callbacks import EarlyStopping
from biovoice.training.checkpointing import save_checkpoint
from biovoice.training.optimization import build_optimizer
from biovoice.training.trainer import fit_model
from biovoice.utils.serialization import save_json


def _spoof_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["spoof_label"].to(outputs["logits"].device)
    return spoof_classification_loss(outputs["logits"], labels)


def _spoof_accuracy(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> float:
    labels = batch["spoof_label"].to(outputs["logits"].device)
    predictions = (outputs["probability"] >= 0.5).float()
    return float((predictions == labels).float().mean().item())


def train_spoof_baseline(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train the baseline anti-spoof model and save checkpoints/history."""
    utterance_manifest = load_manifest(config["data"]["utterance_manifest_path"])
    validation_split = config["data"]["validation_split"]
    if (utterance_manifest["split"] == validation_split).sum() == 0:
        validation_split = config["data"]["train_split"]
    train_dataset = UtteranceDataset(
        utterance_manifest,
        preprocessing_config=config["preprocessing"],
        split=config["data"]["train_split"],
    )
    val_dataset = UtteranceDataset(
        utterance_manifest,
        preprocessing_config=config["preprocessing"],
        split=validation_split,
    )
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_utterances)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_utterances)

    model = build_anti_spoof_model(config)
    optimizer = build_optimizer(model, config)
    history = fit_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        device=config["training"]["device"],
        epochs=int(config["training"]["epochs"]),
        loss_fn=_spoof_loss,
        metric_fn=_spoof_accuracy,
        early_stopping=EarlyStopping(int(config["training"]["early_stopping_patience"])),
    )
    checkpoint_path = run_dir / "checkpoints" / "spoof_model.pt"
    save_checkpoint(checkpoint_path, model)
    history_payload = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "train_metric": history.train_metric,
        "val_metric": history.val_metric,
    }
    save_json(history_payload, run_dir / "reports" / "spoof_history.json")
    pd.DataFrame(history_payload).to_csv(run_dir / "tables" / "spoof_history.csv", index=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history_payload,
    }
