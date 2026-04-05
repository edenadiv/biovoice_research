"""Plots for dataset inspection and preprocessing review.

These utilities intentionally favor legibility over stylistic minimalism because
their main audience is a supervisor reviewing the experimental protocol.
"""

from __future__ import annotations

import pandas as pd

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_class_balance(
    frame: pd.DataFrame,
    path: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    """Plot the class distribution for a manifest or prediction table."""
    plt = prepare_figure(style)
    counts = frame["label"].value_counts().sort_index()
    counts.plot(kind="bar", color=["#457b9d", "#e76f51", "#2a9d8f"][: len(counts)])
    plt.title("Class Balance")
    plt.xlabel("Class")
    plt.ylabel("Count")
    save_current_figure(path, dpi=dpi)


def plot_duration_histogram(
    frame: pd.DataFrame,
    path: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
    title: str = "Probe Duration Histogram",
) -> None:
    """Plot an interpretable duration histogram for the chosen table."""
    plt = prepare_figure(style)
    frame["duration_seconds"].hist(bins=12, color="#457b9d")
    plt.title(title)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    save_current_figure(path, dpi=dpi)


def plot_numeric_histogram(
    frame: pd.DataFrame,
    column: str,
    path: str,
    title: str,
    xlabel: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
    bins: int = 12,
) -> None:
    """Plot a histogram for a numeric quality or metadata column."""
    plt = prepare_figure(style)
    frame[column].hist(bins=bins, color="#6c757d")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_current_figure(path, dpi=dpi)
