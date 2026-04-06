"""Generic training loops for the baseline branch models."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass(slots=True)
class History:
    """Training history saved for later plotting."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metric: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    best_monitor_value: float | None = None
    monitor_name: str | None = None


def run_epoch(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor],
    metric_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], float],
) -> tuple[float, float]:
    """Run one train or evaluation epoch."""
    is_training = optimizer is not None
    model.train(is_training)
    losses: list[float] = []
    metrics: list[float] = []
    for batch in dataloader:
        inputs = batch["waveform"].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, batch)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        metrics.append(float(metric_fn(outputs, batch)))
    return float(sum(losses) / max(len(losses), 1)), float(sum(metrics) / max(len(metrics), 1))


def fit_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    loss_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor],
    metric_fn: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], float],
    early_stopping=None,
    monitor_name: str = "val_loss",
    restore_best_state: bool = False,
) -> History:
    """Train a model and track loss/metric curves."""
    history = History()
    model.to(device)
    history.monitor_name = monitor_name
    mode = getattr(early_stopping, "mode", "min") if early_stopping is not None else ("max" if monitor_name == "val_metric" else "min")
    best_value = float("inf") if mode == "min" else float("-inf")
    best_epoch: int | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    for epoch_index in range(epochs):
        train_loss, train_metric = run_epoch(model, train_loader, optimizer, device, loss_fn, metric_fn)
        val_loss, val_metric = run_epoch(model, val_loader, None, device, loss_fn, metric_fn)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_metric.append(train_metric)
        history.val_metric.append(val_metric)
        monitored_value = val_metric if monitor_name == "val_metric" else val_loss
        improved = monitored_value < best_value if mode == "min" else monitored_value > best_value
        if improved:
            best_value = monitored_value
            best_epoch = epoch_index
            if restore_best_state:
                best_state_dict = copy.deepcopy(model.state_dict())
        if early_stopping is not None and early_stopping.step(monitored_value):
            break
    if restore_best_state and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    history.best_epoch = best_epoch
    history.best_monitor_value = None if best_epoch is None else float(best_value)
    return history
