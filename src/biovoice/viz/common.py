"""Shared matplotlib helpers."""

from __future__ import annotations

import os
from pathlib import Path

cache_dir = Path(__file__).resolve().parents[3] / ".matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_figure(style: str = "seaborn-v0_8-whitegrid"):
    """Apply the repository plot style and return pyplot."""
    plt.style.use(style)
    return plt


def save_current_figure(path: str | Path, dpi: int = 180) -> None:
    """Save the current figure with consistent layout."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close()
