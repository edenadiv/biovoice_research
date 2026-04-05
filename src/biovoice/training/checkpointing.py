"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, extra_state: dict[str, Any] | None = None) -> None:
    """Save a model checkpoint plus any auxiliary state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "extra_state": extra_state or {}}
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load model weights and return the stored auxiliary state."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return payload.get("extra_state", {})
