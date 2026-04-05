"""Tests for the metadata-driven private-corpus staging workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from biovoice.data.manifests import load_manifest
from biovoice.utils.audio_io import save_audio
from biovoice.workflows import prepare_data_workflow, run_joint_workflow


def _base_waveform(frequency: float, duration_seconds: float = 1.4, sample_rate: int = 16000) -> torch.Tensor:
    time = torch.linspace(0.0, duration_seconds, steps=int(duration_seconds * sample_rate))
    signal = 0.35 * torch.sin(2 * torch.pi * frequency * time)
    signal += 0.12 * torch.sin(2 * torch.pi * 2.0 * frequency * time)
    signal += 0.01 * torch.randn_like(signal)
    return signal.unsqueeze(0).clamp(-1.0, 1.0)


def _spoof_waveform(waveform: torch.Tensor) -> torch.Tensor:
    signal = waveform.squeeze(0)
    shifted = torch.roll(signal, shifts=90)
    spoof = 0.8 * shifted + 0.03 * torch.sign(torch.sin(torch.linspace(0.0, 60.0, steps=signal.numel())))
    return spoof.unsqueeze(0).clamp(-1.0, 1.0)


def _write_private_corpus_fixture(root: Path, include_splits: bool = False, overlapping_split: bool = False) -> Path:
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    speakers = [f"speaker_{index:02d}" for index in range(5)]
    split_map = {
        "speaker_00": "train",
        "speaker_01": "train",
        "speaker_02": "val",
        "speaker_03": "test",
        "speaker_04": "test",
    }
    for speaker_index, speaker_id in enumerate(speakers):
        speaker_dir = audio_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        frequency = 140.0 + 20.0 * speaker_index
        for utterance_index in range(5):
            source_id = f"{speaker_id}_source_{utterance_index:02d}"
            bona = _base_waveform(frequency + utterance_index)
            bona_path = speaker_dir / f"{source_id}_bona.wav"
            save_audio(bona_path, bona, 16000)
            row = {
                "utterance_id": f"{speaker_id}_bona_{utterance_index:02d}",
                "speaker_id": speaker_id,
                "relative_path": str(bona_path.relative_to(root)),
                "spoof_label": 0,
                "source_recording_id": source_id,
            }
            if include_splits:
                row["split"] = "test" if overlapping_split and speaker_id == "speaker_00" and utterance_index >= 3 else split_map[speaker_id]
            rows.append(row)
        for utterance_index in range(2):
            source_id = f"{speaker_id}_spoof_source_{utterance_index:02d}"
            bona = _base_waveform(frequency + 0.5 * utterance_index)
            spoof = _spoof_waveform(bona)
            spoof_path = speaker_dir / f"{source_id}_spoof.wav"
            save_audio(spoof_path, spoof, 16000)
            row = {
                "utterance_id": f"{speaker_id}_spoof_{utterance_index:02d}",
                "speaker_id": speaker_id,
                "relative_path": str(spoof_path.relative_to(root)),
                "spoof_label": 1,
                "source_recording_id": source_id,
            }
            if include_splits:
                row["split"] = split_map[speaker_id]
            rows.append(row)
    metadata_path = root / "raw_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    return metadata_path


def _private_corpus_config(tmp_path: Path, metadata_path: Path) -> dict[str, object]:
    corpus_root = metadata_path.parent
    manifest_root = corpus_root / "staged_manifests"
    return {
        "experiment": {"name": "private_corpus_smoke", "mode": "fusion_plus_interpretable_features", "seed": 11, "output_root": str(tmp_path / "runs")},
        "data": {
            "source_type": "real_private_corpus",
            "dataset_name": "fixture_private_corpus",
            "dataset_root": str(corpus_root),
            "raw_metadata_path": str(metadata_path),
            "manifest_output_dir": str(manifest_root),
            "utterance_manifest_path": str(manifest_root / "utterances.csv"),
            "trial_manifest_path": str(manifest_root / "trials.csv"),
            "split_manifest_dir": str(manifest_root / "splits"),
            "sample_rate": 16000,
            "train_split": "train",
            "validation_split": "val",
            "test_split": "test",
            "max_duration_seconds": 3.0,
            "min_duration_seconds": 1.0,
            "enrollment_count": 2,
            "speaker_disjoint": True,
            "require_speaker_disjoint": True,
            "split_strategy": "speaker_disjoint",
            "use_existing_splits": False,
            "probe_trials_per_speaker": 1,
            "wrong_speaker_trials_per_speaker": 2,
            "impostor_sampling_strategy": "round_robin",
            "speech_threshold": 0.02,
            "quality_scan_mode": "full",
            "quality_waveform_sample_size": 4,
            "quality_progress_every": 10,
        },
        "preprocessing": {
            "target_sample_rate": 16000,
            "mono": True,
            "loudness_normalize": True,
            "silence_trim": True,
            "silence_threshold": 0.02,
            "min_speech_ratio": 0.2,
            "pad_to_seconds": 3.0,
            "truncate_to_seconds": 3.0,
        },
        "segmentation": {"window_seconds": 1.0, "hop_seconds": 0.5, "min_segment_seconds": 0.5},
        "model": {
            "feature": {"n_mels": 32, "n_fft": 256, "hop_length": 128, "win_length": 256},
            "speaker_encoder": {"embedding_dim": 32, "hidden_channels": 16},
            "anti_spoof": {"hidden_channels": 16},
            "fusion": {"use_trainable_head": True, "include_interpretable_features": True},
        },
        "training": {"device": "cpu", "batch_size": 2, "epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0, "early_stopping_patience": 1},
        "evaluation": {"sv_threshold": 0.5, "spoof_threshold": 0.5, "manual_review_margin": 0.0, "calibration_bins": 5, "threshold_sweep_points": 5},
        "plotting": {"dpi": 100, "style": "seaborn-v0_8-whitegrid"},
        "explainability": {"suspicious_segment_count": 2, "explanation_feature_count": 3},
        "ablation": {"enabled": True, "segment_windows": [1.0], "overlaps": [0.5]},
    }


def test_private_corpus_prepare_stages_manifests_and_reports(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_root = Path(prepare_data_workflow(config_path))
    manifest_root = Path(config["data"]["manifest_output_dir"])
    assert (manifest_root / "utterances.csv").exists()
    assert (manifest_root / "trials.csv").exists()
    assert (manifest_root / "quality_summary.csv").exists()
    assert (manifest_root / "dataset_summary.json").exists()
    assert (run_root / "tables" / "leakage_report.csv").exists()
    assert (run_root / "plots" / "speech_ratio_histogram.png").exists()


def test_private_corpus_prepare_validates_required_columns(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    metadata = pd.read_csv(metadata_path).drop(columns=["source_recording_id"])
    metadata.to_csv(metadata_path, index=False)
    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_missing_column.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_data_workflow(config_path)


def test_private_corpus_prepare_normalizes_string_spoof_labels_and_absolute_paths(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    metadata = pd.read_csv(metadata_path)
    absolute_paths = []
    for _, row in metadata.iterrows():
        resolved = (metadata_path.parent / str(row["relative_path"])).resolve()
        absolute_paths.append(str(resolved))
    metadata["path"] = absolute_paths
    metadata = metadata.drop(columns=["relative_path"])
    metadata["spoof_label"] = metadata["spoof_label"].map({0: "bona_fide", 1: "spoof"})
    metadata.to_csv(metadata_path, index=False)

    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_absolute_paths.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepare_data_workflow(config_path)
    utterances = load_manifest(config["data"]["utterance_manifest_path"])
    assert set(utterances["spoof_label"].unique()) == {0, 1}
    assert all(Path(path).is_absolute() for path in utterances["path"].tolist())


def test_private_corpus_prepare_filters_short_audio(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    metadata = pd.read_csv(metadata_path)
    short_waveform = _base_waveform(180.0, duration_seconds=0.25)
    short_path = metadata_path.parent / "audio" / "speaker_00" / "speaker_00_short.wav"
    save_audio(short_path, short_waveform, 16000)
    metadata = pd.concat(
        [
            metadata,
            pd.DataFrame(
                [
                    {
                        "utterance_id": "speaker_00_short",
                        "speaker_id": "speaker_00",
                        "relative_path": str(short_path.relative_to(metadata_path.parent)),
                        "spoof_label": 0,
                        "source_recording_id": "speaker_00_short_source",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    metadata.to_csv(metadata_path, index=False)

    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_short_filter.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepare_data_workflow(config_path)
    utterances = load_manifest(config["data"]["utterance_manifest_path"])
    assert "speaker_00_short" not in set(utterances["utterance_id"].tolist())


@pytest.mark.parametrize(
    ("scan_mode", "expected_values"),
    [
        ("full", {"waveform"}),
        ("header_only", {"header_only"}),
        ("header_plus_sample", {"header_only", "waveform"}),
    ],
)
def test_private_corpus_quality_scan_modes(tmp_path: Path, scan_mode: str, expected_values: set[str]) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    config = _private_corpus_config(tmp_path, metadata_path)
    config["data"]["quality_scan_mode"] = scan_mode
    config["data"]["quality_waveform_sample_size"] = 6
    config_path = tmp_path / f"private_{scan_mode}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepare_data_workflow(config_path)
    quality_frame = pd.read_csv(Path(config["data"]["manifest_output_dir"]) / "quality_summary.csv")
    assert expected_values.issubset(set(quality_frame["quality_measurement"].unique()))
    if scan_mode == "header_only":
        assert quality_frame["speech_ratio"].isna().all()
    if scan_mode == "header_plus_sample":
        assert quality_frame["speech_ratio"].notna().any()
        assert quality_frame["speech_ratio"].isna().any()


def test_private_corpus_prepare_rejects_split_overlap(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture", include_splits=True, overlapping_split=True)
    config = _private_corpus_config(tmp_path, metadata_path)
    config["data"]["use_existing_splits"] = True
    config_path = tmp_path / "private_overlap.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    try:
        prepare_data_workflow(config_path)
    except ValueError as error:
        assert "Speaker-disjoint" in str(error)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("Expected speaker-disjoint validation to fail.")


@pytest.mark.parametrize("strategy", ["round_robin", "seeded_shuffle"])
def test_private_corpus_wrong_speaker_trial_multiplicity(tmp_path: Path, strategy: str) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    config = _private_corpus_config(tmp_path, metadata_path)
    config["data"]["wrong_speaker_trials_per_speaker"] = 3
    config["data"]["impostor_sampling_strategy"] = strategy
    config_path = tmp_path / f"private_wrong_speaker_{strategy}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    prepare_data_workflow(config_path)
    trials = load_manifest(config["data"]["trial_manifest_path"])
    wrong = trials[(trials["split"] == "test") & (trials["label"] == "wrong_speaker")]
    counts = wrong.groupby("speaker_id").size().to_dict()
    assert counts
    assert set(counts.values()) == {3}


def test_private_corpus_prepare_rejects_when_target_trials_disappear_after_leakage_filter(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    metadata = pd.read_csv(metadata_path)
    mask = metadata["speaker_id"].isin(["speaker_03", "speaker_04"]) & (metadata["spoof_label"] == 0)
    metadata.loc[mask, "source_recording_id"] = metadata.loc[mask, "speaker_id"].astype(str) + "_shared_source"
    metadata.to_csv(metadata_path, index=False)

    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_target_overlap.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required trial labels"):
        prepare_data_workflow(config_path)


def test_private_corpus_joint_workflow_smoke(tmp_path: Path) -> None:
    metadata_path = _write_private_corpus_fixture(tmp_path / "fixture")
    config = _private_corpus_config(tmp_path, metadata_path)
    config_path = tmp_path / "private_joint.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_root = Path(run_joint_workflow(config_path))
    assert (run_root / "metrics.json").exists()
    assert (run_root / "reports" / "dataset_review.json").exists()
    assert (run_root / "reports" / "supervisor_report.md").exists()
    assert (run_root / "plots" / "reliability_diagram.png").exists()
