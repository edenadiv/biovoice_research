"""Optimizer and scheduler helpers."""

from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build the default Adam optimizer."""
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
