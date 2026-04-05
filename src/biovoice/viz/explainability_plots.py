"""Plots for suspicious segments and explanation cues."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from biovoice.viz.common import prepare_figure, save_current_figure


def plot_waveform_with_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    suspicious_segments: pd.DataFrame,
    path: str,
    dpi: int = 180,
    style: str = "seaborn-v0_8-whitegrid",
) -> None:
    plt = prepare_figure(style)
    signal = waveform.squeeze().detach().cpu().numpy()
    time_axis = np.arange(signal.shape[-1]) / sample_rate
    plt.plot(time_axis, signal, linewidth=1.0, color="#264653")
    for _, row in suspicious_segments.iterrows():
        plt.axvspan(row["start_seconds"], row["end_seconds"], color="#e76f51", alpha=0.25)
    plt.title("Waveform With Suspicious Segments")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    save_current_figure(path, dpi=dpi)


def plot_segment_score_timeline(segment_frame: pd.DataFrame, column: str, title: str, path: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    plt = prepare_figure(style)
    centers = 0.5 * (segment_frame["start_seconds"] + segment_frame["end_seconds"])
    plt.plot(centers, segment_frame[column], marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel(column.replace("_", " ").title())
    save_current_figure(path, dpi=dpi)


def plot_feature_contributions(top_features: list[dict], path: str, dpi: int = 180, style: str = "seaborn-v0_8-whitegrid") -> None:
    plt = prepare_figure(style)
    labels = [item["feature"] for item in top_features]
    values = [float(item["value"]) for item in top_features]
    plt.barh(labels, values, color="#457b9d")
    plt.title("Top Feature Contribution Magnitudes")
    plt.xlabel("Absolute Delta")
    save_current_figure(path, dpi=dpi)
