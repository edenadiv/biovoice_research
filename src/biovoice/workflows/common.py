"""Shared workflow helpers used across data prep, training, and reporting.

The repository's orchestration layer needs a small amount of shared logic:
configuration normalization, run-directory setup, audit summary generation, and
artifact inventory exports. Keeping those pieces here prevents the larger
workflow entry points from becoming another monolith.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biovoice.data.quality_checks import leakage_overlap_report, speaker_split_report, summarize_audio_quality
from biovoice.reports.artifact_inventory import build_artifact_index, build_plot_inventory, flatten_metric_dict
from biovoice.reports.table_export import export_table
from biovoice.utils.audio_io import inspect_audio_metadata, load_audio
from biovoice.utils.config_utils import load_config, save_yaml
from biovoice.utils.logging_utils import configure_logging
from biovoice.utils.path_utils import RunPaths, create_run_paths, resolve_path


ALPHA_LIMITATIONS = [
    "Baselines are research baselines, not production systems.",
    "No real-world robustness claims should be made until broader evaluation is completed.",
    "Explainability outputs are supporting evidence, not proof of causality.",
    "Short or noisy audio may weaken both speaker-verification and spoof judgments.",
    "Stronger pretrained backbones and larger-scale benchmarking are future work.",
]

METRIC_INTERPRETATION_NOTES = {
    "sv": "Treat SV metrics as biometric-consistency evidence only. Strong spoofers can still fool this branch.",
    "spoof": "Treat spoof metrics as artifact-detection evidence only. Bona fide wrong speakers can still appear non-spoofed.",
    "joint": "Joint metrics matter most for the research question because they reflect the final three-way decision.",
    "calibration": "Calibration metrics matter when thresholds and probabilities are used in reports or manual review rules.",
}


def load_workflow_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and fill in the path defaults used by every workflow."""
    config = load_config(config_path)
    data_cfg = config.setdefault("data", {})
    data_cfg.setdefault("source_type", "demo")
    data_cfg.setdefault("dataset_name", data_cfg.get("source_type", "dataset"))
    if "manifest_output_dir" not in data_cfg:
        if data_cfg.get("source_type") == "demo":
            data_cfg["manifest_output_dir"] = str(resolve_path(data_cfg.get("demo_root", "demo_data")) / "manifests")
        elif data_cfg.get("source_type") == "asvspoof2021_la":
            data_cfg["manifest_output_dir"] = "external_data/asvspoof/staged_manifests"
        elif data_cfg.get("utterance_manifest_path"):
            data_cfg["manifest_output_dir"] = str(Path(data_cfg["utterance_manifest_path"]).resolve().parent)
    if data_cfg.get("manifest_output_dir"):
        manifest_root = resolve_path(data_cfg["manifest_output_dir"])
        data_cfg.setdefault("utterance_manifest_path", str(manifest_root / "utterances.csv"))
        data_cfg.setdefault("trial_manifest_path", str(manifest_root / "trials.csv"))
        data_cfg.setdefault("split_manifest_dir", str(manifest_root / "splits"))
    data_cfg.setdefault("split_strategy", "speaker_disjoint")
    data_cfg.setdefault("use_existing_splits", False)
    data_cfg.setdefault("require_speaker_disjoint", bool(data_cfg.get("speaker_disjoint", True)))
    data_cfg.setdefault("quality_scan_mode", "full")
    data_cfg.setdefault("quality_progress_every", 500)
    config.setdefault("training", {}).setdefault("device", "auto")
    return config


def setup_run(config: dict[str, Any], experiment_name: str) -> tuple[RunPaths, Any]:
    """Create the standard run directory and save a config snapshot."""
    run_paths = create_run_paths(config["experiment"]["output_root"], experiment_name)
    save_yaml(config, run_paths.configs / "config_snapshot.yaml")
    logger = configure_logging(run_paths.logs / "run.log")
    return run_paths, logger


def sample_quality_subset(
    utterances: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Choose a deterministic audit subset for expensive waveform quality scans."""
    if sample_size <= 0 or sample_size >= len(utterances):
        return utterances.copy()
    group_columns = [column for column in ["split", "spoof_label"] if column in utterances.columns]
    if not group_columns:
        group_columns = ["spoof_label"] if "spoof_label" in utterances.columns else []
    if not group_columns:
        return utterances.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    grouped = utterances.groupby(group_columns, dropna=False)
    sampled_parts: list[pd.DataFrame] = []
    for _, part in grouped:
        proportional = max(1, int(round(len(part) / len(utterances) * sample_size)))
        sampled_parts.append(part.sample(n=min(len(part), proportional), random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["utterance_id"])
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def compute_quality_frame(
    utterances: pd.DataFrame,
    speech_threshold: float,
    *,
    scan_mode: str = "full",
    waveform_sample_size: int | None = None,
    seed: int = 42,
    logger: Any | None = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Compute quality summaries using either full or sampled waveform scans."""
    scan_mode = str(scan_mode).lower()
    rows: list[dict[str, Any]] = []
    waveform_subset = utterances
    if scan_mode == "header_only":
        waveform_subset = utterances.iloc[0:0]
    elif scan_mode == "header_plus_sample":
        sample_size = int(waveform_sample_size or 0)
        waveform_subset = sample_quality_subset(utterances, sample_size=sample_size, seed=seed)
    elif scan_mode != "full":
        raise ValueError(
            "Unsupported quality scan mode. Expected one of: "
            "'full', 'header_only', 'header_plus_sample'."
        )

    waveform_ids = set(waveform_subset["utterance_id"].tolist())
    if scan_mode != "full":
        if logger is not None:
            logger.info(
                "Quality scan mode '%s': waveform summaries for %d/%d utterances.",
                scan_mode,
                len(waveform_subset),
                len(utterances),
            )
        for _, row in utterances.iterrows():
            metadata = inspect_audio_metadata(row["path"])
            duration_seconds = float(metadata["num_frames"] / max(metadata["sample_rate"], 1))
            measurement = "waveform" if row["utterance_id"] in waveform_ids else "header_only"
            rows.append(
                {
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "path": row["path"],
                    "spoof_label": int(row["spoof_label"]),
                    "source_recording_id": row.get("source_recording_id", ""),
                    "duration_seconds": duration_seconds,
                    "speech_ratio": np.nan,
                    "sample_rate": int(metadata["sample_rate"]),
                    "clipping_ratio": np.nan,
                    "peak_amplitude": np.nan,
                    "rms": np.nan,
                    "snr_proxy_db": np.nan,
                    "quality_measurement": measurement,
                }
            )
        rows_by_id = {str(item["utterance_id"]): item for item in rows}
        for index, (_, row) in enumerate(waveform_subset.iterrows(), start=1):
            waveform, sample_rate = load_audio(row["path"])
            summary = summarize_audio_quality(waveform, sample_rate, threshold=speech_threshold)
            rows_by_id[str(row["utterance_id"])].update(summary.to_dict())
            if logger is not None and index % max(progress_every, 1) == 0:
                logger.info(
                    "Computed waveform quality for %d/%d sampled utterances.",
                    index,
                    len(waveform_subset),
                )
        return pd.DataFrame(rows)

    for index, (_, row) in enumerate(utterances.iterrows(), start=1):
        waveform, sample_rate = load_audio(row["path"])
        summary = summarize_audio_quality(waveform, sample_rate, threshold=speech_threshold)
        rows.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "path": row["path"],
                "spoof_label": int(row["spoof_label"]),
                "source_recording_id": row.get("source_recording_id", ""),
                **summary.to_dict(),
                "quality_measurement": "waveform",
            }
        )
        if logger is not None and index % max(progress_every, 1) == 0:
            logger.info("Computed waveform quality for %d/%d utterances.", index, len(utterances))
    return pd.DataFrame(rows)


def dataset_review_summary(
    config: dict[str, Any],
    utterances: pd.DataFrame,
    trials: pd.DataFrame,
    quality_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a compact dataset-review payload for reports and notebooks."""
    data_cfg = config["data"]
    leak_report = leakage_overlap_report(trials)
    split_report = speaker_split_report(utterances)
    summary: dict[str, Any] = {
        "dataset_mode": str(data_cfg.get("source_type", "demo")),
        "dataset_name": str(data_cfg.get("dataset_name", "dataset")),
        "manifest_output_dir": str(resolve_path(data_cfg["manifest_output_dir"])),
        "split_strategy": str(data_cfg.get("split_strategy", "speaker_disjoint")),
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
        "speaker_disjoint_violations": int(split_report["violates_speaker_disjoint"].sum()),
        "trial_leakage_violations": int(leak_report["has_leakage"].sum()),
    }
    summary["speaker_disjoint_status"] = "pass" if summary["speaker_disjoint_violations"] == 0 else "fail"
    summary["trial_leakage_status"] = "pass" if summary["trial_leakage_violations"] == 0 else "fail"
    if "dataset_root" in data_cfg:
        summary["dataset_root"] = str(resolve_path(data_cfg["dataset_root"]))
    if "raw_metadata_path" in data_cfg:
        summary["raw_metadata_path"] = str(resolve_path(data_cfg["raw_metadata_path"]))
    if quality_frame is not None and not quality_frame.empty:
        summary["mean_duration_seconds"] = float(quality_frame["duration_seconds"].dropna().mean())
        if quality_frame["speech_ratio"].notna().any():
            summary["mean_speech_ratio"] = float(quality_frame["speech_ratio"].dropna().mean())
    return summary


def load_staged_dataset_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Load the staged dataset summary when data prep already saved one."""
    summary_path = resolve_path(config["data"]["manifest_output_dir"]) / "dataset_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def merge_dataset_review(
    config: dict[str, Any],
    utterances: pd.DataFrame,
    trials: pd.DataFrame,
    quality_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Merge staged dataset metadata with run-time review summaries."""
    cached = load_staged_dataset_summary(config)
    summary = {
        **cached,
        **dataset_review_summary(config, utterances, trials, quality_frame=quality_frame),
    }
    summary["quality_scan_mode"] = str(summary.get("quality_scan_mode", config["data"].get("quality_scan_mode", "full")))
    if quality_frame is not None and "quality_measurement" in quality_frame.columns:
        measurement_counts = {
            key: int(value)
            for key, value in quality_frame["quality_measurement"].value_counts(dropna=False).sort_index().items()
        }
        summary["quality_measurement_counts"] = measurement_counts
        summary["waveform_scanned_files"] = int(measurement_counts.get("waveform", 0))
        summary["header_only_files"] = int(measurement_counts.get("header_only", 0))
    return summary


def build_alpha_checklist(run_paths: RunPaths) -> dict[str, bool]:
    """Create a concrete alpha-exit checklist from saved artifacts."""
    required_plots = [
        "class_balance.png",
        "duration_histogram.png",
        "sv_loss_curves.png",
        "spoof_loss_curves.png",
        "sv_roc.png",
        "spoof_roc.png",
        "sv_pr.png",
        "spoof_pr.png",
        "sv_det.png",
        "spoof_det.png",
        "confusion_matrix.png",
        "target_vs_non_target_scores.png",
        "spoof_vs_bonafide_scores.png",
        "reliability_diagram.png",
        "ablation_summary.png",
        "waveform_with_suspicious_segments.png",
        "spoof_score_over_time.png",
        "speaker_similarity_over_time.png",
    ]
    return {
        "End-to-end run produced metrics.json": (run_paths.root / "metrics.json").exists(),
        "Predictions table was saved": (run_paths.root / "predictions.csv").exists(),
        "Mandatory plots were generated": all((run_paths.plots / name).exists() for name in required_plots),
        "Supervisor report was generated": (run_paths.reports / "supervisor_report.md").exists(),
        "Artifact index was generated": (run_paths.tables / "artifact_index.csv").exists(),
        "Plot inventory was generated": (run_paths.tables / "plot_inventory.csv").exists(),
    }


def save_inventory_tables(run_paths: RunPaths, metrics: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save artifact, plot, and metric index tables for supervisor browsing."""
    artifact_index = build_artifact_index(run_paths.root)
    plot_inventory = build_plot_inventory(run_paths.root)
    metric_summary = flatten_metric_dict(metrics)
    export_table(artifact_index, run_paths.tables / "artifact_index.csv", run_paths.reports / "artifact_index.md")
    export_table(plot_inventory, run_paths.tables / "plot_inventory.csv", run_paths.reports / "plot_inventory.md")
    export_table(metric_summary, run_paths.tables / "metric_summary.csv", run_paths.reports / "metric_summary.md")
    return artifact_index, plot_inventory, metric_summary
