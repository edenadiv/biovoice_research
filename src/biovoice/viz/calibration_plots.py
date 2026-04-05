"""Calibration and reliability plots."""

from __future__ import annotations

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_reliability_diagram(summary: dict, path: str, title: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    plt = prepare_figure(style)
    plt.plot(summary["prob_pred"], summary["prob_true"], marker="o", linewidth=2, label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    save_current_figure(path, dpi=dpi)
