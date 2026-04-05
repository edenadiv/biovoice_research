"""Training curve plots."""

from __future__ import annotations

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_loss_curves(history: dict, path: str, title: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    plt = prepare_figure(style)
    plt.plot(history["train_loss"], label="Train loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_current_figure(path, dpi=dpi)
