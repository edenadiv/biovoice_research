"""Speaker verification baseline training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from biovoice.data.loading import UtteranceDataset, collate_utterances
from biovoice.data.manifests import load_manifest
from biovoice.models.losses import speaker_classification_loss
from biovoice.models.model_factory import build_speaker_model
from biovoice.training.callbacks import EarlyStopping
from biovoice.training.checkpointing import save_checkpoint
from biovoice.training.device import resolve_device
from biovoice.training.optimization import build_optimizer
from biovoice.training.trainer import fit_model
from biovoice.utils.serialization import save_json


def _speaker_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["speaker_index"].to(outputs["logits"].device)
    return speaker_classification_loss(outputs["logits"], labels)


def _speaker_accuracy(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> float:
    labels = batch["speaker_index"].to(outputs["logits"].device)
    predictions = outputs["logits"].argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def train_speaker_baseline(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Train the baseline speaker model and save checkpoints/history."""
    device = resolve_device(config["training"].get("device", "auto"))
    utterance_manifest = load_manifest(config["data"]["utterance_manifest_path"])
    train_frame = utterance_manifest[utterance_manifest["split"] == config["data"]["train_split"]]
    train_frame = train_frame[train_frame["spoof_label"] == 0].reset_index(drop=True)
    speakers = sorted(train_frame["speaker_id"].unique())
    speaker_to_index = {speaker: index for index, speaker in enumerate(speakers)}

    train_mask = train_frame.groupby("speaker_id").cumcount() % 5 != 0
    train_subset = train_frame[train_mask].reset_index(drop=True)
    val_subset = train_frame[~train_mask].reset_index(drop=True)
    if val_subset.empty:
        val_subset = train_subset.copy()

    train_dataset = UtteranceDataset(
        train_subset,
        preprocessing_config=config["preprocessing"],
        split=config["data"]["train_split"],
        speaker_to_index=speaker_to_index,
        only_bona_fide=True,
    )
    val_dataset = UtteranceDataset(
        val_subset,
        preprocessing_config=config["preprocessing"],
        split=config["data"]["train_split"],
        speaker_to_index=speaker_to_index,
        only_bona_fide=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_utterances)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_utterances)

    model = build_speaker_model(config, num_speakers=len(speakers))
    optimizer = build_optimizer(model, config)
    history = fit_model(
        model,
        train_loader,
        val_loader,
        optimizer=optimizer,
        device=device,
        epochs=int(config["training"]["epochs"]),
        loss_fn=_speaker_loss,
        metric_fn=_speaker_accuracy,
        early_stopping=EarlyStopping(int(config["training"]["early_stopping_patience"])),
    )

    checkpoint_path = run_dir / "checkpoints" / "speaker_model.pt"
    save_checkpoint(checkpoint_path, model, extra_state={"speaker_to_index": speaker_to_index})
    history_payload = {
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "train_metric": history.train_metric,
        "val_metric": history.val_metric,
    }
    save_json(history_payload, run_dir / "reports" / "sv_history.json")
    pd.DataFrame(history_payload).to_csv(run_dir / "tables" / "sv_history.csv", index=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history": history_payload,
        "speaker_to_index": speaker_to_index,
    }
