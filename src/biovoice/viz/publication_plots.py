"""Compact comparison plots for model summaries and ablations."""

from __future__ import annotations

import pandas as pd

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_comparison_bars(frame: pd.DataFrame, x: str, y: str, title: str, path: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    plt = prepare_figure(style)
    plt.bar(frame[x], frame[y], color="#2a9d8f")
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.xticks(rotation=20, ha="right")
    save_current_figure(path, dpi=dpi)


def plot_grouped_metric_bars(
    frame: pd.DataFrame,
    *,
    x: str,
    metrics: list[str],
    title: str,
    path: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Plot several evaluation metrics side-by-side for each baseline mode."""
    plt = prepare_figure(style)
    x_values = frame[x].tolist()
    bar_positions = range(len(x_values))
    width = 0.22 if len(metrics) >= 3 else 0.35
    palette = ["#2a9d8f", "#457b9d", "#e9c46a", "#e76f51"]
    offsets = [((index - (len(metrics) - 1) / 2.0) * width) for index in range(len(metrics))]
    for metric_index, metric in enumerate(metrics):
        positions = [position + offsets[metric_index] for position in bar_positions]
        plt.bar(
            positions,
            frame[metric],
            width=width,
            color=palette[metric_index % len(palette)],
            label=metric.replace("_", " ").title(),
        )
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel("Score")
    plt.xticks(list(bar_positions), x_values, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.legend()
    save_current_figure(path, dpi=dpi)
