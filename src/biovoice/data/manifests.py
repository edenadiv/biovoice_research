"""Manifest schemas and validation utilities.

The project uses explicit manifests because enrollment-conditioned experiments
are vulnerable to subtle leakage. Inspectable CSV/JSON files make the protocol
auditable for supervisors and paper writing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_TRIAL_COLUMNS = {
    "trial_id",
    "speaker_id",
    "probe_path",
    "enrollment_paths",
    "label",
    "split",
}
REQUIRED_UTTERANCE_COLUMNS = {
    "utterance_id",
    "speaker_id",
    "path",
    "split",
    "spoof_label",
}


@dataclass(slots=True)
class TrialRecord:
    """Canonical representation of one enrollment-conditioned verification trial."""

    trial_id: str
    speaker_id: str
    probe_path: str
    enrollment_paths: list[str]
    label: str
    split: str
    claim_id: str | None = None
    source_recording_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a manifest-friendly dictionary."""
        payload = asdict(self)
        payload["enrollment_paths"] = "|".join(self.enrollment_paths)
        return payload


def _normalize_manifest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "enrollment_paths" in frame.columns:
        frame["enrollment_paths"] = frame["enrollment_paths"].apply(
            lambda value: value if isinstance(value, list) else str(value).split("|")
        )
    return frame


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON manifest and normalize list-valued columns."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        frame = pd.read_json(source)
    else:
        frame = pd.read_csv(source)
    return _normalize_manifest_columns(frame)


def save_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist a manifest to CSV or JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    if "enrollment_paths" in serializable.columns:
        serializable["enrollment_paths"] = serializable["enrollment_paths"].apply(
            lambda value: "|".join(value) if isinstance(value, list) else value
        )
    if target.suffix.lower() == ".json":
        serializable.to_json(target, orient="records", indent=2)
    else:
        serializable.to_csv(target, index=False)


def validate_trial_manifest(frame: pd.DataFrame) -> None:
    """Validate a trial manifest and raise descriptive errors when invalid."""
    missing = REQUIRED_TRIAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Trial manifest missing required columns: {sorted(missing)}")
    if frame["trial_id"].duplicated().any():
        raise ValueError("Trial manifest contains duplicated trial_id values.")
    if frame["enrollment_paths"].apply(lambda value: len(value) == 0).any():
        raise ValueError("Every trial must include at least one enrollment file.")


def validate_utterance_manifest(frame: pd.DataFrame) -> None:
    """Validate the utterance manifest used for branch-specific training."""
    missing = REQUIRED_UTTERANCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Utterance manifest missing required columns: {sorted(missing)}")
    if frame["utterance_id"].duplicated().any():
        raise ValueError("Utterance manifest contains duplicated utterance_id values.")


def save_split_manifests(frame: pd.DataFrame, output_dir: str | Path, prefix: str) -> None:
    """Save one manifest file per split to make leakage review straightforward."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    for split, part in frame.groupby("split"):
        save_manifest(part.reset_index(drop=True), base / f"{prefix}_{split}.csv")
