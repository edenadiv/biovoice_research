"""Common serialization helpers for JSON, CSV, and pandas tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with indentation for supervisor readability."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_frame(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Persist a pandas DataFrame as CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=index)
