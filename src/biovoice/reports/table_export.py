"""CSV and Markdown table export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a simple Markdown table without extra dependencies."""
    if frame.empty:
        return "_No rows available._"
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def export_table(frame: pd.DataFrame, csv_path: str | Path, markdown_path: str | Path | None = None) -> None:
    """Save a DataFrame as CSV and optionally Markdown."""
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_target, index=False)
    if markdown_path is not None:
        target = Path(markdown_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(dataframe_to_markdown(frame), encoding="utf-8")
