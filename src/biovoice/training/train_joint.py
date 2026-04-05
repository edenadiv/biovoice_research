"""Late fusion training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biovoice.models.fusion_model import LateFusionModel
from biovoice.models.losses import fusion_classification_loss
from biovoice.models.model_factory import build_fusion_head
from biovoice.training.device import resolve_device
from biovoice.training.optimization import build_optimizer
from biovoice.utils.serialization import save_json


LABEL_TO_INDEX = {"wrong_speaker": 0, "spoof": 1, "target_bona_fide": 2, "manual_review": 3}


def train_fusion_head(config: dict[str, Any], train_frame: pd.DataFrame, val_frame: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    """Train a small fusion head over saved branch scores."""
    device = resolve_device(config["training"].get("device", "auto"))
    feature_columns = [column for column in train_frame.columns if column.startswith("fusion_feature_")]
    if not feature_columns:
        raise ValueError("Fusion training requires feature columns prefixed with 'fusion_feature_'.")
    model = build_fusion_head(input_dim=len(feature_columns))
    model.to(device)
    optimizer = build_optimizer(model, config)

    x_train = torch.tensor(train_frame[feature_columns].to_numpy(dtype=np.float32), device=device)
    y_train = torch.tensor(train_frame["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64), device=device)
    x_val = torch.tensor(val_frame[feature_columns].to_numpy(dtype=np.float32), device=device)
    y_val = torch.tensor(val_frame["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64), device=device)

    history = {"train_loss": [], "val_loss": []}
    model.train()
    for _ in range(int(config["training"]["epochs"])):
        logits = model(x_train)
        loss = fusion_classification_loss(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = fusion_classification_loss(model(x_val), y_val)
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss.item()))

    checkpoint_path = run_dir / "checkpoints" / "fusion_head.pt"
    torch.save({"state_dict": model.state_dict(), "feature_columns": feature_columns}, checkpoint_path)
    save_json(history, run_dir / "reports" / "fusion_history.json")
    return {"checkpoint_path": str(checkpoint_path), "feature_columns": feature_columns, "history": history}


def apply_rule_fusion(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic late fusion baseline."""
    model = LateFusionModel()
    result = frame.copy()
    result["fusion_score"] = [
        model(sv, spoof, delta)
        for sv, spoof, delta in zip(
            result["sv_score"],
            result["spoof_probability"],
            result["global_feature_abs_delta_mean"].fillna(0.0),
        )
    ]
    return result
