"""Tests for baseline comparison, error summaries, and spoof-loss helpers."""

from __future__ import annotations

import pandas as pd
import torch

from biovoice.models.losses import build_spoof_loss
from biovoice.training.callbacks import EarlyStopping
from biovoice.training.trainer import fit_model
from biovoice.workflows.evaluation import (
    build_baseline_comparison_frame,
    build_decision_path_summary_frame,
    build_error_summary_frame,
)


def _prediction_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [
                "spoof",
                "spoof",
                "target_bona_fide",
                "wrong_speaker",
                "wrong_speaker",
            ],
            "sv_score": [0.1, 0.8, 0.9, 0.2, 0.7],
            "spoof_probability": [0.9, 0.7, 0.2, 0.2, 0.3],
            "majority_spoof_decision": ["spoof"] * 5,
            "sv_only_decision": [
                "wrong_speaker",
                "target_bona_fide",
                "target_bona_fide",
                "wrong_speaker",
                "target_bona_fide",
            ],
            "spoof_only_decision": ["spoof", "spoof", "target_bona_fide", "target_bona_fide", "target_bona_fide"],
            "fusion_default_decision": ["spoof", "spoof", "target_bona_fide", "wrong_speaker", "target_bona_fide"],
            "fusion_tuned_decision": ["spoof", "spoof", "wrong_speaker", "wrong_speaker", "wrong_speaker"],
            "final_decision": ["spoof", "spoof", "wrong_speaker", "wrong_speaker", "wrong_speaker"],
            "spoof_label": [1, 1, 0, 0, 0],
        }
    )


def test_baseline_comparison_includes_majority_and_fusion_modes() -> None:
    comparison = build_baseline_comparison_frame(_prediction_fixture())
    assert {
        "majority_spoof",
        "sv_only",
        "spoof_only",
        "fusion_default",
        "fusion_tuned",
    }.issubset(set(comparison["mode"]))


def test_error_summary_highlights_off_diagonal_confusions() -> None:
    error_summary = build_error_summary_frame(_prediction_fixture())
    assert not error_summary.empty
    assert set(error_summary.columns) == {"true_label", "predicted_label", "count", "rate_within_true_label"}


def test_decision_path_summary_counts_spoof_gate_and_sv_paths() -> None:
    summary = build_decision_path_summary_frame(_prediction_fixture(), sv_threshold=0.5, spoof_threshold=0.5)
    assert set(summary["path"]) == {"spoof_gate", "sv_accept_after_spoof_reject", "sv_reject_after_spoof_reject"}


def test_weighted_spoof_loss_builds_and_runs() -> None:
    loss_fn = build_spoof_loss(loss_name="weighted_bce", pos_weight=2.0)
    logits = torch.tensor([0.2, -0.3], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    loss = loss_fn(logits, labels)
    assert float(loss.item()) > 0.0


def test_early_stopping_supports_max_mode() -> None:
    callback = EarlyStopping(patience=2, mode="max")
    assert callback.step(0.1) is False
    assert callback.step(0.2) is False
    assert callback.step(0.19) is False
    assert callback.step(0.18) is True


def test_fit_model_restores_best_state_for_metric_monitor(monkeypatch) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        def forward(self, inputs):  # pragma: no cover - patched run_epoch bypasses this
            return {"logits": self.weight + inputs, "probability": torch.sigmoid(self.weight)}

    # The patched epoch runner simulates three epochs whose validation metric
    # peaks on the second epoch while the model parameter keeps drifting.
    train_weights = [0.0, 1.0, 2.0]
    val_metrics = [0.3, 0.8, 0.5]
    val_losses = [1.0, 0.9, 0.95]
    call_state = {"epoch": 0}

    def fake_run_epoch(model, dataloader, optimizer, device, loss_fn, metric_fn):
        epoch = call_state["epoch"]
        if optimizer is not None:
            model.weight.data.fill_(train_weights[epoch])
            return (1.0 - 0.1 * epoch, 0.1 * epoch)
        result = (val_losses[epoch], val_metrics[epoch])
        call_state["epoch"] += 1
        return result

    monkeypatch.setattr("biovoice.training.trainer.run_epoch", fake_run_epoch)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = fit_model(
        model,
        train_loader=[object()],
        val_loader=[object()],
        optimizer=optimizer,
        device="cpu",
        epochs=3,
        loss_fn=lambda outputs, batch: torch.tensor(0.0),
        metric_fn=lambda outputs, batch: 0.0,
        early_stopping=EarlyStopping(patience=3, mode="max"),
        monitor_name="val_metric",
        restore_best_state=True,
    )
    assert history.best_epoch == 1
    assert history.best_monitor_value == 0.8
    assert history.monitor_name == "val_metric"
    assert torch.isclose(model.weight.detach().cpu(), torch.tensor([1.0], dtype=torch.float32)).all()
