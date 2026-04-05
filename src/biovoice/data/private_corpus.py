"""Metadata-driven private corpus staging for real-data experiments.

This adapter is intentionally conservative. Its job is to turn a user-managed
table of local audio files into the repository's canonical manifests while
making leakage assumptions explicit and auditable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biovoice.data.manifests import save_manifest, save_split_manifests, validate_trial_manifest, validate_utterance_manifest
from biovoice.data.quality_checks import (
    assert_no_trial_leakage,
    assert_speaker_disjoint,
    speaker_split_report,
    summarize_audio_quality,
)
from biovoice.utils.audio_io import inspect_audio_metadata, load_audio
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


def _stable_rank(*parts: object, seed: int) -> int:
    """Create a deterministic ordering key independent of filesystem ordering."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(f"{seed}|{payload}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


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


def _sample_quality_subset(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Choose a deterministic audit subset for costly waveform quality scans."""
    if sample_size <= 0 or sample_size >= len(frame):
        return frame.copy()
    group_columns = [column for column in ["split", "spoof_label"] if column in frame.columns]
    if not group_columns:
        group_columns = ["spoof_label"]
    grouped = frame.groupby(group_columns, dropna=False)
    sampled_parts: list[pd.DataFrame] = []
    for _, part in grouped:
        proportional = max(1, int(round(len(part) / len(frame) * sample_size)))
        sampled_parts.append(part.sample(n=min(len(part), proportional), random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["utterance_id"])
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def _quality_frame(
    frame: pd.DataFrame,
    speech_threshold: float,
    *,
    scan_mode: str,
    waveform_sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Compute scalable quality summaries used for filtering and review artifacts."""
    scan_mode = str(scan_mode).lower()
    rows: list[dict[str, Any]] = []
    waveform_subset = frame
    if scan_mode == "header_only":
        waveform_subset = frame.iloc[0:0]
    elif scan_mode == "header_plus_sample":
        waveform_subset = _sample_quality_subset(frame, waveform_sample_size, seed)
    elif scan_mode != "full":
        raise ValueError(
            "Unsupported quality_scan_mode. Expected one of "
            "'full', 'header_only', or 'header_plus_sample'."
        )

    waveform_ids = set(waveform_subset["utterance_id"].tolist())
    if scan_mode != "full":
        for _, row in frame.iterrows():
            metadata = inspect_audio_metadata(row["path"])
            duration_seconds = float(metadata["num_frames"] / max(metadata["sample_rate"], 1))
            rows.append(
                {
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "path": row["path"],
                    "source_recording_id": row["source_recording_id"],
                    "spoof_label": row["spoof_label"],
                    "duration_seconds": duration_seconds,
                    "speech_ratio": np.nan,
                    "sample_rate": int(metadata["sample_rate"]),
                    "clipping_ratio": np.nan,
                    "peak_amplitude": np.nan,
                    "rms": np.nan,
                    "snr_proxy_db": np.nan,
                    "quality_measurement": "waveform" if row["utterance_id"] in waveform_ids else "header_only",
                }
            )
        rows_by_id = {str(item["utterance_id"]): item for item in rows}
        for _, row in waveform_subset.iterrows():
            waveform, sample_rate = load_audio(row["path"])
            summary = summarize_audio_quality(waveform, sample_rate, threshold=speech_threshold)
            rows_by_id[str(row["utterance_id"])].update(summary.to_dict())
        return pd.DataFrame(rows)

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
                "quality_measurement": "waveform",
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


def _rank_frame(frame: pd.DataFrame, seed: int, *context: object) -> pd.DataFrame:
    """Attach a stable pseudo-random order to reduce file-order bias."""
    ranked = frame.copy()
    ranked["_selection_rank"] = ranked.apply(
        lambda row: _stable_rank(
            *context,
            row["speaker_id"],
            row["source_recording_id"],
            row["utterance_id"],
            seed=seed,
        ),
        axis=1,
    )
    return ranked.sort_values(["_selection_rank", "source_recording_id", "utterance_id"]).reset_index(drop=True)


def _select_enrollment_rows(
    bona: pd.DataFrame,
    enrollment_count: int,
    seed: int,
    split: str,
    speaker_id: str,
) -> pd.DataFrame:
    """Choose deterministic enrollment utterances with less file-order bias.

    Old behavior selected the first files in sorted order, which could bias the
    enrollment pool toward a particular recording or naming pattern. The new
    policy uses a seed-stable ranking and prefers one utterance per
    ``source_recording_id`` before reusing additional recordings.
    """
    ranked = _rank_frame(bona, seed, "enrollment", split, speaker_id)
    chosen_indices: list[int] = []
    for _, group in ranked.groupby("source_recording_id", sort=False):
        chosen_indices.append(int(group.index[0]))
        if len(chosen_indices) == enrollment_count:
            break
    if len(chosen_indices) < enrollment_count:
        for index in ranked.index:
            if int(index) not in chosen_indices:
                chosen_indices.append(int(index))
            if len(chosen_indices) == enrollment_count:
                break
    return ranked.loc[chosen_indices].sort_values("_selection_rank").drop(columns="_selection_rank")


def _select_probe_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
    split: str,
    speaker_id: str,
    role: str,
    limit: int | None,
) -> pd.DataFrame:
    """Pick deterministic probe candidates without inheriting filename order."""
    ranked = _rank_frame(frame, seed, role, split, speaker_id)
    if limit is None:
        return ranked.drop(columns="_selection_rank")
    return ranked.head(limit).drop(columns="_selection_rank")


def _select_wrong_speaker_rows(
    split_frame: pd.DataFrame,
    *,
    claimed_speaker_id: str,
    enrollment_sources: set[str],
    split: str,
    seed: int,
    count: int,
    strategy: str,
) -> pd.DataFrame:
    """Build deterministic, leakage-safe wrong-speaker probes for one claim."""
    eligible = split_frame[
        (split_frame["speaker_id"] != claimed_speaker_id)
        & (split_frame["spoof_label"] == 0)
        & (~split_frame["source_recording_id"].isin(enrollment_sources))
    ].copy()
    if eligible.empty or count <= 0:
        return eligible.iloc[0:0]

    strategy = str(strategy).lower()
    if strategy == "seeded_shuffle":
        return _rank_frame(eligible, seed, "wrong-speaker", split, claimed_speaker_id).head(count).drop(columns="_selection_rank")
    if strategy != "round_robin":
        raise ValueError("Unsupported impostor_sampling_strategy. Expected 'round_robin' or 'seeded_shuffle'.")

    other_speakers = sorted(eligible["speaker_id"].astype(str).unique())
    speaker_order = sorted(
        other_speakers,
        key=lambda speaker: _stable_rank("wrong-speaker-speaker", split, claimed_speaker_id, speaker, seed=seed),
    )
    selected_parts: list[pd.DataFrame] = []
    per_speaker_ranked = {
        speaker: _rank_frame(
            eligible[eligible["speaker_id"] == speaker].copy(),
            seed,
            "wrong-speaker-probe",
            split,
            claimed_speaker_id,
            speaker,
        ).reset_index(drop=True)
        for speaker in speaker_order
    }
    cursor = 0
    while len(selected_parts) < count:
        progress = False
        for speaker in speaker_order:
            ranked = per_speaker_ranked[speaker]
            if cursor < len(ranked):
                selected_parts.append(ranked.iloc[[cursor]])
                progress = True
                if len(selected_parts) == count:
                    break
        if not progress:
            break
        cursor += 1
    if not selected_parts:
        return eligible.iloc[0:0]
    return pd.concat(selected_parts, ignore_index=True).drop(columns="_selection_rank")


def _build_trial_manifest(utterances: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    """Construct leakage-safe enrollment-conditioned trials from utterances."""
    enrollment_count = int(data_cfg["enrollment_count"])
    probe_limit = data_cfg.get("probe_trials_per_speaker")
    probe_limit = None if probe_limit in {None, "", 0} else int(probe_limit)
    wrong_speaker_trials = int(data_cfg.get("wrong_speaker_trials_per_speaker", 1))
    impostor_strategy = str(data_cfg.get("impostor_sampling_strategy", "round_robin"))
    seed = int(data_cfg.get("seed", 42))
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
            bona = speaker_frame[speaker_frame["spoof_label"] == 0].copy()
            spoof = speaker_frame[speaker_frame["spoof_label"] == 1].copy()
            if len(bona) < enrollment_count + 1:
                raise ValueError(
                    f"Speaker {speaker_id} in split {split} needs at least "
                    f"{enrollment_count + 1} bona fide utterances."
                )
            enrollment = _select_enrollment_rows(
                bona,
                enrollment_count=enrollment_count,
                seed=seed,
                split=str(split),
                speaker_id=str(speaker_id),
            )
            remaining_bona = bona.loc[~bona["utterance_id"].isin(set(enrollment["utterance_id"]))].copy()
            target_candidates = _select_probe_rows(
                remaining_bona,
                seed=seed,
                split=str(split),
                speaker_id=str(speaker_id),
                role="target_probe",
                limit=probe_limit,
            )
            spoof_candidates = _select_probe_rows(
                spoof,
                seed=seed,
                split=str(split),
                speaker_id=str(speaker_id),
                role="spoof_probe",
                limit=probe_limit,
            )
            enrollment_paths = enrollment["path"].tolist()
            enrollment_sources = enrollment["source_recording_id"].astype(str).tolist()
            enrollment_source_set = set(enrollment_sources)
            target_candidates = target_candidates[
                ~target_candidates["source_recording_id"].astype(str).isin(enrollment_source_set)
            ].reset_index(drop=True)

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

            for _, probe in spoof_candidates.iterrows():
                if str(probe["source_recording_id"]) in enrollment_source_set:
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

            impostor_candidates = _select_wrong_speaker_rows(
                split_frame,
                claimed_speaker_id=str(speaker_id),
                enrollment_sources=enrollment_source_set,
                split=str(split),
                seed=seed,
                count=wrong_speaker_trials,
                strategy=impostor_strategy,
            )
            for _, probe in impostor_candidates.iterrows():
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
        "mean_speech_ratio": float(quality_frame["speech_ratio"].dropna().mean()) if quality_frame["speech_ratio"].notna().any() else None,
        "speaker_disjoint_violations": int(speaker_report["violates_speaker_disjoint"].sum()),
        "quality_scan_mode": str(data_cfg.get("quality_scan_mode", "full")),
        "quality_measurement_counts": {
            key: int(value)
            for key, value in quality_frame["quality_measurement"].value_counts(dropna=False).sort_index().items()
        },
        "waveform_scanned_files": int((quality_frame["quality_measurement"] == "waveform").sum()),
        "header_only_files": int((quality_frame["quality_measurement"] == "header_only").sum()),
        "enrollment_policy_summary": (
            "Enrollment utterances are chosen with a seed-stable ranking that prefers distinct "
            "source recordings before reusing additional bona fide files."
        ),
        "wrong_speaker_policy_summary": (
            f"{int(data_cfg.get('wrong_speaker_trials_per_speaker', 1))} wrong-speaker probe(s) per claimed speaker "
            f"using {str(data_cfg.get('impostor_sampling_strategy', 'round_robin'))} impostor sampling."
        ),
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
    data_cfg["seed"] = int(config["experiment"]["seed"])

    quality = _quality_frame(
        metadata,
        speech_threshold=float(data_cfg["speech_threshold"]),
        scan_mode=str(data_cfg.get("quality_scan_mode", "full")),
        waveform_sample_size=int(data_cfg.get("quality_waveform_sample_size", 0) or 0),
        seed=int(config["experiment"]["seed"]),
    )
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
