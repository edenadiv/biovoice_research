"""ASVspoof 2019/2021 LA staging helpers.

This module implements one concrete real-data path that is aligned with the
project's research question:

- train the spoof and speaker branches on ASVspoof 2019 LA train/dev
- build enrollment pools from ASVspoof 2019 LA eval bona fide audio
- evaluate probes on ASVspoof 2021 LA evaluation audio

The implementation intentionally uses only official protocol files and official
metadata mappings so the resulting manifests can be audited by supervisors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import tarfile
import zipfile
from typing import Any

import pandas as pd

from biovoice.data.manifests import (
    save_manifest,
    save_split_manifests,
    validate_trial_manifest,
    validate_utterance_manifest,
)
from biovoice.data.quality_checks import assert_no_trial_leakage, speaker_split_report
from biovoice.utils.path_utils import resolve_path
from biovoice.utils.serialization import save_frame, save_json


@dataclass(slots=True)
class ASVspoofPaths:
    """Resolved input and staging paths for the ASVspoof adapter."""

    raw_2019_zip: Path
    raw_2021_eval_tar: Path
    keys_tar: Path
    meta_tar: Path
    extract_root: Path
    manifest_root: Path
    split_root: Path


def _resolve_paths(data_cfg: dict[str, Any]) -> ASVspoofPaths:
    """Resolve configured archive and output paths."""
    manifest_root = resolve_path(data_cfg["manifest_output_dir"])
    return ASVspoofPaths(
        raw_2019_zip=resolve_path(data_cfg["asvspoof2019_la_zip"]),
        raw_2021_eval_tar=resolve_path(data_cfg["asvspoof2021_la_eval_tar"]),
        keys_tar=resolve_path(data_cfg["asvspoof2021_keys_tar"]),
        meta_tar=resolve_path(data_cfg["asvspoof_meta_tar"]),
        extract_root=resolve_path(data_cfg["extract_root"]),
        manifest_root=manifest_root,
        split_root=resolve_path(data_cfg["split_manifest_dir"]),
    )


def _require_existing_file(path: Path) -> None:
    """Raise a clear error if a configured ASVspoof asset is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required ASVspoof asset not found: {path}")


def _extract_zip_once(source: Path, target_dir: Path) -> Path:
    """Extract a zip archive only when the output directory is absent."""
    destination = target_dir / source.stem
    if destination.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as handle:
        handle.extractall(destination)
    return destination


def _extract_tar_once(source: Path, target_dir: Path) -> Path:
    """Extract a tar archive only when the output directory is absent."""
    stem = source.name
    for suffix in [".tar.gz", ".tgz", ".tar"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    destination = target_dir / stem
    if destination.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source) as handle:
        handle.extractall(destination)
    return destination


def _find_one(root: Path, pattern: str) -> Path:
    """Find exactly one matching file or directory beneath a root."""
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find pattern '{pattern}' under {root}")
    if len(matches) > 1:
        # Prefer the shortest path when duplicates exist inside nested wrappers.
        matches = sorted(matches, key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def _stable_rank(*parts: object, seed: int) -> int:
    """Create a seed-stable ordering key independent of archive file ordering."""
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.blake2b(f"{seed}|{payload}".encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


def _read_table(path: Path, sep: str | None = None, header: int | None = None) -> pd.DataFrame:
    """Read whitespace or tab-separated metadata tables."""
    if sep is None:
        return pd.read_csv(path)
    return pd.read_csv(path, sep=sep, header=header)


def _speaker_from_mapping_row(row: pd.Series) -> str:
    """Derive the target speaker id from an official mapping row."""
    vctk_id = str(row["VCTK_ID"])
    if vctk_id != "-" and "_" in vctk_id:
        return vctk_id.split("_", 1)[0]
    target = str(row["TTS_VC_target_speaker"])
    if target and target != "-":
        return target
    raise ValueError(f"Unable to derive speaker id from mapping row: {row.to_dict()}")


def _recording_id_from_mapping_row(row: pd.Series, asvspoof_id: str) -> str:
    """Choose a stable source-recording identifier for leakage checks."""
    vctk_id = str(row["VCTK_ID"])
    if vctk_id != "-":
        return vctk_id
    vc_source = str(row["VC_source_VCTK_ID"])
    if vc_source != "-":
        return vc_source
    return f"spoof_{asvspoof_id}"


def _load_mapping_table(extracted_meta_root: Path, filename: str) -> pd.DataFrame:
    """Load one official ASVspoof mapping TSV."""
    path = _find_one(extracted_meta_root, filename)
    frame = pd.read_csv(path, sep="\t")
    return frame


def _load_2021_trial_metadata(extracted_keys_root: Path) -> pd.DataFrame:
    """Load the official 2021 LA CM metadata.

    The file format is:
    ``track_or_speaker file_id codec transmission attack label trim subset``
    """
    path = _find_one(extracted_keys_root, "trial_metadata.txt")
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=[
            "claimed_id",
            "file_id",
            "codec",
            "transmission",
            "attack_id",
            "key",
            "trim",
            "subset",
        ],
    )
    return frame


def _load_2019_cm_protocol(protocol_root: Path, split_name: str) -> pd.DataFrame:
    """Load one ASVspoof 2019 LA CM protocol file.

    The expected format is:
    ``speaker_id file_id codec_or_placeholder attack_id label``.
    The exact middle fields vary slightly across protocol releases, so the
    loader keeps only the columns required by this repository.
    """
    pattern = f"ASVspoof2019.LA.cm.{split_name}.*.txt"
    path = _find_one(protocol_root, pattern)
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"Unexpected ASVspoof 2019 protocol format in {path}")
    frame = pd.DataFrame(
        {
            "protocol_speaker_id": raw.iloc[:, 0].astype(str),
            "file_id": raw.iloc[:, 1].astype(str),
            "attack_id": raw.iloc[:, -2].astype(str),
            "key": raw.iloc[:, -1].astype(str),
            "split": split_name,
        }
    )
    return frame


def _manifest_from_2019_split(
    protocol_frame: pd.DataFrame,
    mapping_frame: pd.DataFrame,
    audio_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    """Build canonical utterance rows for a 2019 split."""
    merged = protocol_frame.merge(
        mapping_frame,
        left_on="file_id",
        right_on="ASVspoof_ID",
        how="left",
        validate="one_to_one",
    )
    if merged["ASVspoof_ID"].isna().any():
        missing = merged.loc[merged["ASVspoof_ID"].isna(), "file_id"].head(5).tolist()
        raise ValueError(f"2019 mapping table is missing file ids: {missing}")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        speaker_id = _speaker_from_mapping_row(row)
        file_id = str(row["file_id"])
        audio_path = audio_dir / f"{file_id}.flac"
        if not audio_path.exists():
            audio_path = audio_dir / f"{file_id}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Could not find audio for {file_id} under {audio_dir}")
        rows.append(
            {
                "utterance_id": file_id,
                "speaker_id": speaker_id,
                "path": str(audio_path),
                "split": split_name,
                "spoof_label": 0 if str(row["key"]).lower() in {"bonafide", "bona_fide"} else 1,
                "source_recording_id": _recording_id_from_mapping_row(row, file_id),
            }
        )
    return pd.DataFrame(rows)


def _build_enrollment_catalog(
    eval_2019_utterances: pd.DataFrame,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Create an audit-friendly enrollment catalog from 2019 eval bona fide audio.

    The catalog keeps *all* candidate bona fide utterances per speaker, deduped
    by ``source_recording_id`` and sorted deterministically. A later helper then
    selects a per-probe subset that excludes any source recording shared with
    the current probe. This is stricter than using a fixed per-speaker pool and
    avoids trial leakage when the same text/source recording appears in both
    ASVspoof 2019 and ASVspoof 2021 material.
    """
    bona = eval_2019_utterances[eval_2019_utterances["spoof_label"] == 0].copy()
    pools: dict[str, list[dict[str, str]]] = {}
    for speaker_id, frame in bona.groupby("speaker_id"):
        ranked = frame.copy()
        ranked["_selection_rank"] = ranked.apply(
            lambda row: _stable_rank(
                "asvspoof-enroll",
                speaker_id,
                row["source_recording_id"],
                row["utterance_id"],
                seed=seed,
            ),
            axis=1,
        )
        ranked = ranked.sort_values(["_selection_rank", "source_recording_id", "utterance_id"]).drop_duplicates(
            subset=["source_recording_id"]
        )
        pools[str(speaker_id)] = [
            {
                "path": str(item["path"]),
                "source_recording_id": str(item["source_recording_id"]),
            }
            for _, item in ranked.iterrows()
        ]
    return pools


def _select_enrollment_for_probe(
    enrollment_catalog: dict[str, list[dict[str, str]]],
    speaker_id: str,
    enrollment_count: int,
    seed: int,
    blocked_sources: set[str] | None = None,
) -> list[dict[str, str]]:
    """Pick a leakage-safe enrollment subset for one probe.

    Parameters
    ----------
    enrollment_catalog:
        All candidate bona fide enrollment utterances per speaker.
    speaker_id:
        Claimed speaker whose enrollment should be used for the trial.
    enrollment_count:
        Number of enrollment utterances required by the current protocol.
    blocked_sources:
        Source-recording identifiers that must not appear in the enrollment
        set for this specific probe. In ASVspoof 2021 LA this is usually just
        the probe's own ``source_recording_id``.
    """
    blocked_sources = blocked_sources or set()
    if speaker_id not in enrollment_catalog:
        raise KeyError(f"No enrollment candidates available for speaker {speaker_id}.")
    candidates = [
        item
        for item in enrollment_catalog[speaker_id]
        if item["source_recording_id"] not in blocked_sources
    ]
    candidates = sorted(
        candidates,
        key=lambda item: _stable_rank(
            "asvspoof-enroll-probe",
            speaker_id,
            item["source_recording_id"],
            Path(item["path"]).name,
            seed=seed,
        ),
    )
    if len(candidates) < enrollment_count:
        raise ValueError(
            f"Speaker {speaker_id} has only {len(candidates)} leakage-safe enrollment "
            f"utterances after excluding sources {sorted(blocked_sources)}, but "
            f"enrollment_count={enrollment_count}."
        )
    return candidates[:enrollment_count]


def _build_2021_trial_manifest(
    trial_metadata: pd.DataFrame,
    mapping_frame: pd.DataFrame,
    eval_audio_dir: Path,
    enrollment_catalog: dict[str, list[dict[str, str]]],
    enrollment_count: int,
    seed: int,
    test_split_name: str,
) -> pd.DataFrame:
    """Construct enrollment-conditioned trials for ASVspoof 2021 LA eval."""
    merged = trial_metadata.merge(
        mapping_frame,
        left_on="file_id",
        right_on="ASVspoof_ID",
        how="left",
        validate="one_to_one",
    )
    if merged["ASVspoof_ID"].isna().any():
        missing = merged.loc[merged["ASVspoof_ID"].isna(), "file_id"].head(5).tolist()
        raise ValueError(f"2021 mapping table is missing file ids: {missing}")
    eval_rows = merged[merged["subset"] == "eval"].copy()
    rows: list[dict[str, Any]] = []
    trial_counter = 0

    bona_fide_eval = []
    for _, row in eval_rows.iterrows():
        file_id = str(row["file_id"])
        probe_path = eval_audio_dir / f"{file_id}.flac"
        if not probe_path.exists():
            raise FileNotFoundError(f"Could not find 2021 eval audio for {file_id} under {eval_audio_dir}")
        speaker_id = _speaker_from_mapping_row(row)
        if speaker_id not in enrollment_catalog:
            continue
        probe_source = _recording_id_from_mapping_row(row, file_id)
        same_speaker_enrollment = _select_enrollment_for_probe(
            enrollment_catalog,
            speaker_id,
            enrollment_count,
            seed=seed,
            blocked_sources={probe_source},
        )
        enrollment_paths = [item["path"] for item in same_speaker_enrollment]
        enrollment_sources = [item["source_recording_id"] for item in same_speaker_enrollment]
        key = str(row["key"]).lower()
        if key in {"bonafide", "bona_fide"}:
            bona_fide_eval.append(
                {
                    "file_id": file_id,
                    "speaker_id": speaker_id,
                    "probe_path": str(probe_path),
                    "probe_source": probe_source,
                    "enrollment_paths": enrollment_paths,
                    "enrollment_sources": enrollment_sources,
                }
            )
            rows.append(
                {
                    "trial_id": f"trial_{trial_counter:07d}",
                    "speaker_id": speaker_id,
                    "claim_id": speaker_id,
                    "probe_path": str(probe_path),
                    "enrollment_paths": enrollment_paths,
                    "label": "target_bona_fide",
                    "split": test_split_name,
                    "speaker_match_label": 1,
                    "spoof_label": 0,
                    "probe_source_recording_id": probe_source,
                    "enrollment_source_recording_ids": "|".join(enrollment_sources),
                }
            )
            trial_counter += 1
        else:
            rows.append(
                {
                    "trial_id": f"trial_{trial_counter:07d}",
                    "speaker_id": speaker_id,
                    "claim_id": speaker_id,
                    "probe_path": str(probe_path),
                    "enrollment_paths": enrollment_paths,
                    "label": "spoof",
                    "split": test_split_name,
                    "speaker_match_label": 1,
                    "spoof_label": 1,
                    "probe_source_recording_id": probe_source,
                    "enrollment_source_recording_ids": "|".join(enrollment_sources),
                }
            )
            trial_counter += 1

    # Add wrong-speaker trials by reusing bona fide probe files under incorrect claims.
    bona_by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in bona_fide_eval:
        bona_by_speaker.setdefault(row["speaker_id"], []).append(row)
    speaker_ids = sorted(bona_by_speaker)
    if len(speaker_ids) < 2:
        raise ValueError("Need at least two ASVspoof 2021 eval speakers to build wrong-speaker trials.")
    for speaker_id in speaker_ids:
        for row in bona_by_speaker[speaker_id]:
            other_candidates = [candidate for candidate in speaker_ids if candidate != speaker_id]
            wrong_claim = sorted(
                other_candidates,
                key=lambda candidate: _stable_rank(
                    "asvspoof-wrong-claim",
                    speaker_id,
                    row["file_id"],
                    candidate,
                    seed=seed,
                ),
            )[0]
            wrong_enrollment = _select_enrollment_for_probe(
                enrollment_catalog,
                wrong_claim,
                enrollment_count,
                seed=seed,
                blocked_sources={str(row["probe_source"])},
            )
            rows.append(
                {
                    "trial_id": f"trial_{trial_counter:07d}",
                    "speaker_id": wrong_claim,
                    "claim_id": wrong_claim,
                    "probe_path": row["probe_path"],
                    "enrollment_paths": [item["path"] for item in wrong_enrollment],
                    "label": "wrong_speaker",
                    "split": test_split_name,
                    "speaker_match_label": 0,
                    "spoof_label": 0,
                    "probe_source_recording_id": row["probe_source"],
                    "enrollment_source_recording_ids": "|".join(
                        item["source_recording_id"] for item in wrong_enrollment
                    ),
                }
            )
            trial_counter += 1

    trial_frame = pd.DataFrame(rows)
    validate_trial_manifest(trial_frame)
    assert_no_trial_leakage(trial_frame)
    return trial_frame


def _build_within_split_trial_manifest(
    split_utterances: pd.DataFrame,
    *,
    split_name: str,
    enrollment_count: int,
    seed: int,
    probe_limit: int | None = None,
) -> pd.DataFrame:
    """Construct validation trials from one bona-fide/spoof split.

    This is used for threshold tuning on ASVspoof 2019 LA dev. It mirrors the
    real evaluation structure in a conservative way:

    - target trials are bona fide probes from the claimed speaker
    - spoof trials are spoof probes under the claimed speaker
    - wrong-speaker trials reuse bona fide probes under a seed-stable wrong claim
    """
    catalog = _build_enrollment_catalog(split_utterances, seed=seed)
    bona = split_utterances[split_utterances["spoof_label"] == 0].copy()
    spoof = split_utterances[split_utterances["spoof_label"] == 1].copy()
    rows: list[dict[str, Any]] = []
    trial_counter = 0
    bona_targets: list[dict[str, Any]] = []
    speaker_ids = sorted(bona["speaker_id"].astype(str).unique())
    for speaker_id in speaker_ids:
        speaker_bona = bona[bona["speaker_id"] == speaker_id].copy()
        speaker_spoof = spoof[spoof["speaker_id"] == speaker_id].copy()
        ranked_bona = speaker_bona.assign(
            _selection_rank=speaker_bona.apply(
                lambda row: _stable_rank(
                    "asvspoof-target-probe",
                    split_name,
                    speaker_id,
                    row["source_recording_id"],
                    row["utterance_id"],
                    seed=seed,
                ),
                axis=1,
            )
        ).sort_values(["_selection_rank", "source_recording_id", "utterance_id"])
        ranked_spoof = speaker_spoof.assign(
            _selection_rank=speaker_spoof.apply(
                lambda row: _stable_rank(
                    "asvspoof-spoof-probe",
                    split_name,
                    speaker_id,
                    row["source_recording_id"],
                    row["utterance_id"],
                    seed=seed,
                ),
                axis=1,
            )
        ).sort_values(["_selection_rank", "source_recording_id", "utterance_id"])
        if probe_limit is not None:
            ranked_bona = ranked_bona.head(probe_limit)
            ranked_spoof = ranked_spoof.head(probe_limit)

        for _, probe in ranked_bona.iterrows():
            probe_source = str(probe["source_recording_id"])
            enrollment = _select_enrollment_for_probe(
                catalog,
                speaker_id,
                enrollment_count,
                seed=seed,
                blocked_sources={probe_source},
            )
            enrollment_paths = [item["path"] for item in enrollment]
            enrollment_sources = [item["source_recording_id"] for item in enrollment]
            rows.append(
                {
                    "trial_id": f"{split_name}_trial_{trial_counter:07d}",
                    "speaker_id": speaker_id,
                    "claim_id": speaker_id,
                    "probe_path": str(probe["path"]),
                    "enrollment_paths": enrollment_paths,
                    "label": "target_bona_fide",
                    "split": split_name,
                    "speaker_match_label": 1,
                    "spoof_label": 0,
                    "probe_source_recording_id": probe_source,
                    "enrollment_source_recording_ids": "|".join(enrollment_sources),
                }
            )
            bona_targets.append(
                {
                    "speaker_id": speaker_id,
                    "probe_path": str(probe["path"]),
                    "probe_source": probe_source,
                    "utterance_id": str(probe["utterance_id"]),
                }
            )
            trial_counter += 1

        for _, probe in ranked_spoof.iterrows():
            probe_source = str(probe["source_recording_id"])
            enrollment = _select_enrollment_for_probe(
                catalog,
                speaker_id,
                enrollment_count,
                seed=seed,
                blocked_sources={probe_source},
            )
            enrollment_paths = [item["path"] for item in enrollment]
            enrollment_sources = [item["source_recording_id"] for item in enrollment]
            rows.append(
                {
                    "trial_id": f"{split_name}_trial_{trial_counter:07d}",
                    "speaker_id": speaker_id,
                    "claim_id": speaker_id,
                    "probe_path": str(probe["path"]),
                    "enrollment_paths": enrollment_paths,
                    "label": "spoof",
                    "split": split_name,
                    "speaker_match_label": 1,
                    "spoof_label": 1,
                    "probe_source_recording_id": probe_source,
                    "enrollment_source_recording_ids": "|".join(enrollment_sources),
                }
            )
            trial_counter += 1

    if len(speaker_ids) < 2:
        raise ValueError("Need at least two speakers to build ASVspoof wrong-speaker validation trials.")
    for target in bona_targets:
        other_candidates = [candidate for candidate in speaker_ids if candidate != target["speaker_id"]]
        wrong_claim = sorted(
            other_candidates,
            key=lambda candidate: _stable_rank(
                "asvspoof-val-wrong-claim",
                split_name,
                target["speaker_id"],
                target["utterance_id"],
                candidate,
                seed=seed,
            ),
        )[0]
        wrong_enrollment = _select_enrollment_for_probe(
            catalog,
            wrong_claim,
            enrollment_count,
            seed=seed,
            blocked_sources={target["probe_source"]},
        )
        rows.append(
            {
                "trial_id": f"{split_name}_trial_{trial_counter:07d}",
                "speaker_id": wrong_claim,
                "claim_id": wrong_claim,
                "probe_path": target["probe_path"],
                "enrollment_paths": [item["path"] for item in wrong_enrollment],
                "label": "wrong_speaker",
                "split": split_name,
                "speaker_match_label": 0,
                "spoof_label": 0,
                "probe_source_recording_id": target["probe_source"],
                "enrollment_source_recording_ids": "|".join(
                    item["source_recording_id"] for item in wrong_enrollment
                ),
            }
        )
        trial_counter += 1

    trial_frame = pd.DataFrame(rows)
    validate_trial_manifest(trial_frame)
    assert_no_trial_leakage(trial_frame)
    return trial_frame


def stage_asvspoof2021_la_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Stage ASVspoof 2019/2021 LA into canonical BioVoice manifests."""
    data_cfg = config["data"]
    paths = _resolve_paths(data_cfg)
    for path in [paths.raw_2019_zip, paths.raw_2021_eval_tar, paths.keys_tar, paths.meta_tar]:
        _require_existing_file(path)
    paths.extract_root.mkdir(parents=True, exist_ok=True)
    paths.manifest_root.mkdir(parents=True, exist_ok=True)
    paths.split_root.mkdir(parents=True, exist_ok=True)

    extracted_2019_root = _extract_zip_once(paths.raw_2019_zip, paths.extract_root)
    extracted_2021_root = _extract_tar_once(paths.raw_2021_eval_tar, paths.extract_root)
    extracted_keys_root = _extract_tar_once(paths.keys_tar, paths.extract_root)
    extracted_meta_root = _extract_tar_once(paths.meta_tar, paths.extract_root)

    protocol_root = _find_one(extracted_2019_root, "ASVspoof2019_LA_cm_protocols")
    train_audio_dir = _find_one(extracted_2019_root, "ASVspoof2019_LA_train")
    dev_audio_dir = _find_one(extracted_2019_root, "ASVspoof2019_LA_dev")
    eval_audio_dir_2019 = _find_one(extracted_2019_root, "ASVspoof2019_LA_eval")
    eval_audio_dir_2021 = _find_one(extracted_2021_root, "ASVspoof2021_LA_eval")

    mapping_2019 = _load_mapping_table(extracted_meta_root, "ASVspoof2019_LA_VCTK_MetaInfo.tsv")
    mapping_2021 = _load_mapping_table(extracted_meta_root, "ASVspoof2021_LA_VCTK_MetaInfo.tsv")
    metadata_2021 = _load_2021_trial_metadata(_find_one(extracted_keys_root, "CM"))

    protocol_train = _load_2019_cm_protocol(protocol_root, "train")
    protocol_dev = _load_2019_cm_protocol(protocol_root, "dev")
    protocol_eval = _load_2019_cm_protocol(protocol_root, "eval")

    utter_train = _manifest_from_2019_split(protocol_train, mapping_2019, train_audio_dir / "flac", "train")
    utter_dev = _manifest_from_2019_split(protocol_dev, mapping_2019, dev_audio_dir / "flac", "val")
    utter_eval = _manifest_from_2019_split(protocol_eval, mapping_2019, eval_audio_dir_2019 / "flac", "enroll")
    utterances = pd.concat([utter_train, utter_dev, utter_eval], ignore_index=True)
    validate_utterance_manifest(utterances)

    if bool(data_cfg.get("speaker_disjoint", True)):
        # ASVspoof 2019 partitions are documented as speaker-disjoint.
        report = speaker_split_report(utterances)
        conflicting = report[(report["violates_speaker_disjoint"]) & (report["splits"] != "enroll")]
        if not conflicting.empty:
            raise ValueError(
                "ASVspoof staged utterances unexpectedly violated speaker-disjoint train/val partitions: "
                f"{conflicting['speaker_id'].tolist()}"
            )

    enrollment_count = int(data_cfg["enrollment_count"])
    selection_seed = int(config.get("experiment", {}).get("seed", 42))
    enrollment_pools = _build_enrollment_catalog(utter_eval, seed=selection_seed)
    insufficient = {
        speaker_id: len(items)
        for speaker_id, items in enrollment_pools.items()
        if len(items) < enrollment_count
    }
    if insufficient:
        raise ValueError(
            "ASVspoof 2019 eval does not provide enough unique-source bona fide "
            f"enrollment candidates for speakers: {insufficient}"
        )
    validation_trials = _build_within_split_trial_manifest(
        utter_dev,
        split_name=str(data_cfg.get("validation_split", "val")),
        enrollment_count=enrollment_count,
        seed=selection_seed,
        probe_limit=None if data_cfg.get("probe_trials_per_speaker") in {None, "", 0} else int(data_cfg["probe_trials_per_speaker"]),
    )
    test_trials = _build_2021_trial_manifest(
        metadata_2021,
        mapping_2021,
        eval_audio_dir_2021 / "flac",
        enrollment_pools,
        enrollment_count=enrollment_count,
        seed=selection_seed,
        test_split_name=str(data_cfg.get("test_split", "test")),
    )
    trials = pd.concat([validation_trials, test_trials], ignore_index=True)

    utterance_manifest = resolve_path(data_cfg["utterance_manifest_path"])
    trial_manifest = resolve_path(data_cfg["trial_manifest_path"])
    save_manifest(utterances, utterance_manifest)
    save_manifest(trials, trial_manifest)
    save_split_manifests(utterances, paths.split_root, "utterances")
    save_split_manifests(trials, paths.split_root, "trials")

    split_report = speaker_split_report(utterances)
    leakage = assert_no_trial_leakage(trials)
    dataset_summary = {
        "dataset_mode": "asvspoof2021_la",
        "dataset_name": str(data_cfg.get("dataset_name", "ASVspoof2021_LA")),
        "protocol_note": "Train audio comes from ASVspoof 2019 LA train. Threshold-tuning validation trials come from ASVspoof 2019 LA dev. Enrollment pools come from ASVspoof 2019 LA eval bona fide. Test probes come from ASVspoof 2021 LA eval.",
        "train_split_note": "ASVspoof 2019 LA train",
        "validation_split_note": "ASVspoof 2019 LA dev trials for threshold tuning",
        "enrollment_pool_note": "ASVspoof 2019 LA eval bona fide",
        "test_split_note": "ASVspoof 2021 LA eval",
        "num_utterances": int(len(utterances)),
        "num_trials": int(len(trials)),
        "num_train_utterances": int((utterances["split"] == "train").sum()),
        "num_val_utterances": int((utterances["split"] == "val").sum()),
        "num_enroll_utterances": int((utterances["split"] == "enroll").sum()),
        "trial_labels": {
            label: int(count)
            for label, count in trials["label"].value_counts().sort_index().items()
        },
        "speaker_split_violations": int(split_report["violates_speaker_disjoint"].sum()),
        "trial_leakage_violations": int(leakage["has_leakage"].sum()),
        "enrollment_policy_summary": (
            "Enrollment candidates are ranked with a seed-stable order over distinct source recordings "
            "rather than relying on archive file order."
        ),
        "wrong_speaker_policy_summary": (
            "Wrong-speaker claims reuse bona fide probes under seed-stable alternative claims while excluding "
            "any enrollment source that overlaps the probe."
        ),
    }
    save_frame(split_report, paths.manifest_root / "speaker_split_report.csv")
    save_frame(leakage, paths.manifest_root / "leakage_report.csv")
    save_json(dataset_summary, paths.manifest_root / "dataset_summary.json")
    return {
        "audio_root": paths.extract_root,
        "utterance_manifest": utterance_manifest,
        "trial_manifest": trial_manifest,
        "split_manifest_dir": paths.split_root,
        "leakage_report": leakage,
        "speaker_split_report": split_report,
        "dataset_summary": dataset_summary,
    }
