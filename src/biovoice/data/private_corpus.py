"""Metadata-driven private corpus staging for real-data experiments.

This adapter is intentionally conservative. Its job is to turn a user-managed
table of local audio files into the repository's canonical manifests while
making leakage assumptions explicit and auditable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from biovoice.data.manifests import save_manifest, save_split_manifests, validate_trial_manifest, validate_utterance_manifest
from biovoice.data.quality_checks import (
    assert_no_trial_leakage,
    assert_speaker_disjoint,
    speaker_split_report,
    summarize_audio_quality,
)
from biovoice.utils.audio_io import load_audio
from biovoice.utils.path_utils import resolve_path
from biovoice.utils.serialization import save_frame, save_json


REQUIRED_PRIVATE_METADATA_COLUMNS = {
    "utterance_id",
    "speaker_id",
    "spoof_label",
    "source_recording_id",
}


def _load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON metadata table for private-corpus staging."""
    source = resolve_path(path)
    if source.suffix.lower() == ".json":
        return pd.read_json(source)
    return pd.read_csv(source)


def _normalize_spoof_label(value: Any) -> int:
    """Convert a flexible spoof label representation into ``0`` or ``1``."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "bona_fide", "bona fide", "bonafide", "human", "genuine", "false"}:
            return 0
        if normalized in {"1", "spoof", "fake", "synthetic", "true"}:
            return 1
        raise ValueError(f"Unsupported spoof_label string value: {value}")
    if pd.isna(value):
        raise ValueError("spoof_label cannot be missing.")
    return int(value)


def _resolve_audio_paths(frame: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """Resolve either absolute ``path`` or repo-relative ``relative_path`` values."""
    resolved = frame.copy()
    if "path" not in resolved.columns and "relative_path" not in resolved.columns:
        raise ValueError("Private corpus metadata must include either 'path' or 'relative_path'.")
    if "path" in resolved.columns:
        resolved["path"] = resolved["path"].apply(lambda value: str(resolve_path(value)))
    else:
        resolved["path"] = resolved["relative_path"].apply(lambda value: str((dataset_root / str(value)).resolve()))
    missing = [path for path in resolved["path"].tolist() if not Path(path).exists()]
    if missing:
        raise ValueError(f"Private corpus metadata references missing audio files: {missing[:5]}")
    return resolved


def _speaker_split_map(speaker_ids: list[str]) -> dict[str, str]:
    """Create a deterministic speaker-disjoint split assignment.

    The rule mirrors the demo corpus: keep at least two speakers in the test
    split so wrong-speaker trials remain constructible.
    """
    ordered = sorted(speaker_ids)
    if len(ordered) < 4:
        raise ValueError(
            "At least four speakers are required for speaker-disjoint real-data "
            "staging so the test split can support wrong-speaker trials."
        )
    test_count = 2
    remaining = len(ordered) - test_count
    val_count = 1 if remaining > 2 else 0
    train_count = len(ordered) - test_count - val_count
    if train_count < 2 and val_count > 0:
        val_count = 0
        train_count = len(ordered) - test_count

    mapping: dict[str, str] = {}
    for speaker_id in ordered[:train_count]:
        mapping[speaker_id] = "train"
    for speaker_id in ordered[train_count : train_count + val_count]:
        mapping[speaker_id] = "val"
    for speaker_id in ordered[train_count + val_count :]:
        mapping[speaker_id] = "test"
    return mapping


def _assign_splits(frame: pd.DataFrame, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    """Apply either provided splits or deterministic speaker-disjoint generation."""
    assigned = frame.copy()
    use_existing = bool(data_cfg.get("use_existing_splits", False))
    has_split_column = "split" in assigned.columns and assigned["split"].notna().all()
    if use_existing and has_split_column:
        assigned["split"] = assigned["split"].astype(str)
        return assigned, True
    if str(data_cfg.get("split_strategy", "speaker_disjoint")) != "speaker_disjoint":
        raise ValueError("The real-data baseline currently supports only the 'speaker_disjoint' split strategy.")
    split_map = _speaker_split_map(sorted(assigned["speaker_id"].astype(str).unique()))
    assigned["split"] = assigned["speaker_id"].map(split_map)
    return assigned, False


def _quality_frame(frame: pd.DataFrame, speech_threshold: float) -> pd.DataFrame:
    """Compute audio quality summaries used for filtering and review artifacts."""
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        waveform, sample_rate = load_audio(row["path"])
        summary = summarize_audio_quality(waveform, sample_rate, threshold=speech_threshold)
        rows.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "path": row["path"],
                "source_recording_id": row["source_recording_id"],
                "spoof_label": row["spoof_label"],
                **summary.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _filter_by_quality(
    frame: pd.DataFrame,
    quality_frame: pd.DataFrame,
    min_duration_seconds: float,
    max_duration_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop files outside the configured duration range and keep merged quality columns."""
    keep_ids = quality_frame.loc[
        (quality_frame["duration_seconds"] >= float(min_duration_seconds))
        & (quality_frame["duration_seconds"] <= float(max_duration_seconds)),
        "utterance_id",
    ]
    filtered = frame[frame["utterance_id"].isin(set(keep_ids))].reset_index(drop=True)
    merged = filtered.merge(
        quality_frame,
        on=["utterance_id", "speaker_id", "path", "source_recording_id", "spoof_label"],
        how="left",
    )
    return filtered, merged


def _build_trial_manifest(utterances: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Construct leakage-safe enrollment-conditioned trials from utterances."""
    enrollment_count = int(data_cfg["enrollment_count"])
    probe_limit = data_cfg.get("probe_trials_per_speaker")
    probe_limit = None if probe_limit in {None, "", 0} else int(probe_limit)
    rows: list[dict[str, Any]] = []
    trial_counter = 0

    for split, split_frame in utterances.groupby("split"):
        speakers = sorted(split_frame["speaker_id"].astype(str).unique())
        if split == str(data_cfg.get("test_split", "test")) and len(speakers) < 2:
            raise ValueError(
                f"Test split '{split}' needs at least two speakers to build wrong-speaker trials."
            )
        for speaker_index, speaker_id in enumerate(speakers):
            speaker_frame = split_frame[split_frame["speaker_id"] == speaker_id].copy()
            bona = speaker_frame[speaker_frame["spoof_label"] == 0].sort_values(
                ["source_recording_id", "utterance_id"]
            )
            spoof = speaker_frame[speaker_frame["spoof_label"] == 1].sort_values(
                ["source_recording_id", "utterance_id"]
            )
            if len(bona) < enrollment_count + 1:
                raise ValueError(
                    f"Speaker {speaker_id} in split {split} needs at least "
                    f"{enrollment_count + 1} bona fide utterances."
                )
            enrollment = bona.iloc[:enrollment_count]
            target_candidates = bona.iloc[enrollment_count:]
            if probe_limit is not None:
                target_candidates = target_candidates.head(probe_limit)
                spoof = spoof.head(probe_limit)
            enrollment_paths = enrollment["path"].tolist()
            enrollment_sources = enrollment["source_recording_id"].astype(str).tolist()

            for _, probe in target_candidates.iterrows():
                rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": probe["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "target_bona_fide",
                        "split": split,
                        "speaker_match_label": 1,
                        "spoof_label": 0,
                        "probe_source_recording_id": probe["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

            for _, probe in spoof.iterrows():
                if str(probe["source_recording_id"]) in enrollment_sources:
                    continue
                rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": probe["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "spoof",
                        "split": split,
                        "speaker_match_label": 1,
                        "spoof_label": 1,
                        "probe_source_recording_id": probe["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

            imposter_candidates = []
            for offset in range(1, len(speakers)):
                imposter_speaker = speakers[(speaker_index + offset) % len(speakers)]
                candidate_frame = split_frame[
                    (split_frame["speaker_id"] == imposter_speaker)
                    & (split_frame["spoof_label"] == 0)
                ].sort_values(["source_recording_id", "utterance_id"])
                if not candidate_frame.empty:
                    imposter_candidates = [candidate_frame.iloc[0]]
                    break
            for probe in imposter_candidates:
                rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": probe["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "wrong_speaker",
                        "split": split,
                        "speaker_match_label": 0,
                        "spoof_label": 0,
                        "probe_source_recording_id": probe["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

    trial_frame = pd.DataFrame(rows)
    validate_trial_manifest(trial_frame)
    assert_no_trial_leakage(trial_frame)
    required_labels = {"target_bona_fide", "spoof", "wrong_speaker"}
    test_split = str(data_cfg.get("test_split", "test"))
    test_labels = set(trial_frame.loc[trial_frame["split"] == test_split, "label"].tolist())
    missing = required_labels - test_labels
    if missing:
        raise ValueError(
            f"Test split '{test_split}' is missing required trial labels: {sorted(missing)}"
        )
    return trial_frame


def _dataset_summary(
    utterances: pd.DataFrame,
    trials: pd.DataFrame,
    quality_frame: pd.DataFrame,
    data_cfg: dict[str, Any],
    used_existing_splits: bool,
) -> dict[str, Any]:
    """Build the dataset-level review payload reused in reports and notebooks."""
    speaker_report = speaker_split_report(utterances)
    return {
        "dataset_mode": "real",
        "dataset_name": str(data_cfg.get("dataset_name", "private_corpus")),
        "dataset_root": str(resolve_path(data_cfg["dataset_root"])),
        "raw_metadata_path": str(resolve_path(data_cfg["raw_metadata_path"])),
        "manifest_output_dir": str(resolve_path(data_cfg["manifest_output_dir"])),
        "split_strategy": str(data_cfg.get("split_strategy", "speaker_disjoint")),
        "used_existing_splits": used_existing_splits,
        "require_speaker_disjoint": bool(data_cfg.get("require_speaker_disjoint", True)),
        "enrollment_count": int(data_cfg["enrollment_count"]),
        "num_utterances": int(len(utterances)),
        "num_trials": int(len(trials)),
        "num_speakers": int(utterances["speaker_id"].nunique()),
        "speakers_per_split": {
            split: int(part["speaker_id"].nunique())
            for split, part in utterances.groupby("split")
        },
        "trials_per_split": {
            split: int(len(part))
            for split, part in trials.groupby("split")
        },
        "trial_labels": {
            label: int(count)
            for label, count in trials["label"].value_counts().sort_index().items()
        },
        "mean_duration_seconds": float(quality_frame["duration_seconds"].mean()),
        "mean_speech_ratio": float(quality_frame["speech_ratio"].mean()),
        "speaker_disjoint_violations": int(speaker_report["violates_speaker_disjoint"].sum()),
    }


def stage_private_corpus_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Stage a local private corpus into canonical BioVoice manifests.

    The function saves canonical manifests and review artifacts directly inside
    the configured manifest output directory so the same staged data can be
    reused by `prepare-data`, training runs, and supervisor notebooks.
    """
    data_cfg = config["data"]
    dataset_root = resolve_path(data_cfg["dataset_root"])
    manifest_root = resolve_path(data_cfg["manifest_output_dir"])
    split_root = resolve_path(data_cfg["split_manifest_dir"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    metadata = _load_table(data_cfg["raw_metadata_path"])
    missing = REQUIRED_PRIVATE_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(
            "Private corpus metadata is missing required columns: "
            f"{sorted(missing)}"
        )
    metadata = _resolve_audio_paths(metadata, dataset_root)
    metadata["speaker_id"] = metadata["speaker_id"].astype(str)
    metadata["utterance_id"] = metadata["utterance_id"].astype(str)
    metadata["source_recording_id"] = metadata["source_recording_id"].astype(str)
    metadata["spoof_label"] = metadata["spoof_label"].apply(_normalize_spoof_label)

    quality = _quality_frame(metadata, speech_threshold=float(data_cfg["speech_threshold"]))
    filtered, quality = _filter_by_quality(
        metadata,
        quality,
        min_duration_seconds=float(data_cfg["min_duration_seconds"]),
        max_duration_seconds=float(data_cfg["max_duration_seconds"]),
    )
    if filtered.empty:
        raise ValueError("All private-corpus utterances were filtered out by duration constraints.")

    with_splits, used_existing_splits = _assign_splits(filtered, data_cfg)
    if bool(data_cfg.get("require_speaker_disjoint", True)):
        assert_speaker_disjoint(with_splits)

    utterances = with_splits[
        [
            "utterance_id",
            "speaker_id",
            "path",
            "split",
            "spoof_label",
            "source_recording_id",
        ]
    ].copy()
    validate_utterance_manifest(utterances)
    trials = _build_trial_manifest(utterances, data_cfg)
    leakage = assert_no_trial_leakage(trials)
    speaker_report = speaker_split_report(utterances)
    summary = _dataset_summary(utterances, trials, quality, data_cfg, used_existing_splits)

    utterance_path = resolve_path(data_cfg["utterance_manifest_path"])
    trial_path = resolve_path(data_cfg["trial_manifest_path"])
    save_manifest(utterances, utterance_path)
    save_manifest(trials, trial_path)
    save_split_manifests(utterances, split_root, "utterances")
    save_split_manifests(trials, split_root, "trials")
    save_frame(quality, manifest_root / "quality_summary.csv")
    save_frame(leakage, manifest_root / "leakage_report.csv")
    save_frame(speaker_report, manifest_root / "speaker_split_report.csv")
    save_json(summary, manifest_root / "dataset_summary.json")

    return {
        "audio_root": dataset_root,
        "utterance_manifest": utterance_path,
        "trial_manifest": trial_path,
        "split_manifest_dir": split_root,
        "quality_frame": quality,
        "leakage_report": leakage,
        "speaker_split_report": speaker_report,
        "dataset_summary": summary,
    }
