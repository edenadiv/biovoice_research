"""Configuration loading and snapshot helpers.

The repository deliberately uses plain YAML instead of a more magical
configuration stack so that supervisors can read the exact experiment settings
without learning a framework-specific override language.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a nested dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dictionaries without mutating the inputs."""
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    """Persist a configuration snapshot for reproducibility."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def load_config(default_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default configuration and optionally apply a user override."""
    config = load_yaml(default_path)
    if override_path:
        override = load_yaml(override_path)
        config = deep_update(config, override)
    return config
