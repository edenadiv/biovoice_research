"""Helpers for repository-relative paths and run output layout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def timestamp_string() -> str:
    """Generate a file-system friendly UTC timestamp."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


@dataclass(slots=True)
class RunPaths:
    """Structured view of a single experiment run directory."""

    root: Path
    logs: Path
    configs: Path
    checkpoints: Path
    plots: Path
    tables: Path
    reports: Path
    explainability: Path
    calibration: Path
    ablations: Path


def create_run_paths(output_root: str | Path, experiment_name: str) -> RunPaths:
    """Create the standard output directory tree for a run."""
    root = resolve_path(output_root) / f"{timestamp_string()}_{experiment_name}"
    root.mkdir(parents=True, exist_ok=True)
    parts = {
        "logs": root / "logs",
        "configs": root / "configs",
        "checkpoints": root / "checkpoints",
        "plots": root / "plots",
        "tables": root / "tables",
        "reports": root / "reports",
        "explainability": root / "explainability",
        "calibration": root / "calibration",
        "ablations": root / "ablations",
    }
    for path in parts.values():
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(root=root, **parts)
