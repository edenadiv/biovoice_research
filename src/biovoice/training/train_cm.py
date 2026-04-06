"""Spoof-detection baseline training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from biovoice.data.loading import UtteranceDataset, collate_utterances
from biovoice.data.manifests import load_manifest
from biovoice.models.losses import build_spoof_loss
from biovoice.models.model_factory import build_anti_spoof_model
from biovoice.training.callbacks import EarlyStopping
from biovoice.training.checkpointing import save_checkpoint
from biovoice.training.device import resolve_device
from biovoice.training.optimization import build_optimizer
from biovoice.training.trainer import fit_model
from biovoice.utils.serialization import save_json


def _spoof_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], loss_fn) -> torch.Tensor:
    labels = batch["spoof_label"].to(outputs["logits"].device)
    return loss_fn(outputs["logits"], labels)


def _spoof_balanced_accuracy(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> float:
    """Balanced accuracy is a better monitor than raw accuracy for spoof skew."""
    labels = batch["spoof_label"].to(outputs["logits"].device)
    predictions = (outputs["probability"] >= 0.5).float()
    positives = labels == 1.0
    negatives = labels == 0.0
    true_positive_rate = float((predictions[positives] == labels[positives]).float().mean().item()) if positives.any() else 0.0
    true_negative_rate = float((predictions[negatives] == labels[negatives]).float().mean().item()) if negatives.any() else 0.0
    if positives.any() and negatives.any():
        return 0.5 * (true_positive_rate + true_negative_rate)
    return true_positive_rate if positives.any() else true_negative_rate


def train_spoof_baseline(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train the baseline anti-spoof model and save checkpoints/history."""
    device = resolve_device(config["training"].get("device", "auto"))
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
    spoof_loss_cfg = config.get("training", {}).get("spoof_loss", {})
    train_frame = utterance_manifest[utterance_manifest["split"] == config["data"]["train_split"]].copy()
    positive_count = max(int(train_frame["spoof_label"].sum()), 1)
    negative_count = max(int((train_frame["spoof_label"] == 0).sum()), 1)
    auto_pos_weight = float(negative_count / positive_count)
    use_pos_weight = bool(spoof_loss_cfg.get("auto_pos_weight", False))
    configured_pos_weight = spoof_loss_cfg.get("positive_class_weight")
    monitor_name = str(config["training"].get("spoof_monitor", "val_metric"))
    if monitor_name not in {"val_loss", "val_metric"}:
        raise ValueError("training.spoof_monitor must be either 'val_loss' or 'val_metric'.")
    monitor_mode = str(
        config["training"].get(
            "spoof_monitor_mode",
            "max" if monitor_name == "val_metric" else "min",
        )
    )
    loss_fn_impl = build_spoof_loss(
        loss_name=str(spoof_loss_cfg.get("name", "bce")),
        pos_weight=float(configured_pos_weight) if configured_pos_weight is not None else (auto_pos_weight if use_pos_weight else None),
        focal_gamma=float(spoof_loss_cfg.get("focal_gamma", 2.0)),
    )
    history = fit_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        device=device,
        epochs=int(config["training"]["epochs"]),
        loss_fn=lambda outputs, batch: _spoof_loss(outputs, batch, loss_fn_impl),
        metric_fn=_spoof_balanced_accuracy,
        early_stopping=EarlyStopping(int(config["training"]["early_stopping_patience"]), mode=monitor_mode),
        monitor_name=monitor_name,
        restore_best_state=bool(config["training"].get("spoof_restore_best_state", True)),
    )
    checkpoint_path = run_dir / "checkpoints" / "spoof_model.pt"
    save_checkpoint(checkpoint_path, model)
    history_payload = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "train_metric": history.train_metric,
        "val_metric": history.val_metric,
        "best_epoch": history.best_epoch,
        "best_monitor_value": history.best_monitor_value,
        "monitor_name": history.monitor_name,
        "monitor_mode": monitor_mode,
    }
    save_json(history_payload, run_dir / "reports" / "spoof_history.json")
    pd.DataFrame(history_payload).to_csv(run_dir / "tables" / "spoof_history.csv", index=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history_payload,
        "loss_name": str(spoof_loss_cfg.get("name", "bce")),
        "monitor_name": monitor_name,
        "monitor_mode": monitor_mode,
        "positive_class_weight": float(configured_pos_weight) if configured_pos_weight is not None else (auto_pos_weight if use_pos_weight else 1.0),
    }
