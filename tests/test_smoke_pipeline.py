"""Minimal smoke test for the synthetic alpha workflow."""

from __future__ import annotations

from pathlib import Path

import yaml

from biovoice.workflows import run_joint_workflow


def test_joint_workflow_smoke(tmp_path: Path) -> None:
    config = {
        "experiment": {"name": "smoke", "mode": "fusion_plus_interpretable_features", "seed": 7, "output_root": str(tmp_path / "runs")},
        "data": {
            "demo_root": str(tmp_path / "demo_data"),
            "utterance_manifest_path": str(tmp_path / "demo_data" / "manifests" / "utterances.csv"),
            "trial_manifest_path": str(tmp_path / "demo_data" / "manifests" / "trials.csv"),
            "split_manifest_dir": str(tmp_path / "demo_data" / "manifests" / "splits"),
            "sample_rate": 16000,
            "train_split": "train",
            "validation_split": "val",
            "test_split": "test",
            "max_duration_seconds": 3.0,
            "min_duration_seconds": 1.0,
            "enrollment_count": 2,
            "speaker_disjoint": True,
            "speech_threshold": 0.02,
            "synthetic_speakers": 4,
            "synthetic_utterances_per_speaker": 4,
            "synthetic_probe_trials_per_speaker": 1,
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
        "evaluation": {"sv_threshold": 0.5, "spoof_threshold": 0.5, "manual_review_margin": 0.0, "calibration_bins": 5},
        "plotting": {"dpi": 100, "style": "seaborn-v0_8-whitegrid"},
        "explainability": {"suspicious_segment_count": 2, "explanation_feature_count": 3},
        "ablation": {"enabled": True, "segment_windows": [1.0], "overlaps": [0.5]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_root = Path(run_joint_workflow(config_path))
    assert (run_root / "metrics.json").exists()
    assert (run_root / "plots" / "confusion_matrix.png").exists()
    assert (run_root / "tables" / "artifact_index.csv").exists()
    assert (run_root / "reports" / "plot_inventory.md").exists()
