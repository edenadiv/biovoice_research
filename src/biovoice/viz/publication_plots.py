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
