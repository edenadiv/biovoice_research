"""Comparison report generation across multiple runs or modes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biovoice.reports.table_export import dataframe_to_markdown


def build_experiment_report(comparison_frame: pd.DataFrame, title: str) -> str:
    """Build a Markdown comparison report."""
    return "\n".join([f"# {title}", "", dataframe_to_markdown(comparison_frame)])


def save_experiment_report(markdown: str, path: str | Path) -> None:
    """Persist the experiment comparison report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
