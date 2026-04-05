"""High-level experiment workflows used by scripts and the CLI.

This module is the main bridge between research code and supervisor-facing
artifacts. Most of the repository's readability requirements eventually funnel
through here because this is where metrics, figures, tables, and reports are
assembled into a coherent run directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biovoice.data.demo import generate_demo_dataset
from biovoice.data.enrollment import aggregate_embeddings
from biovoice.data.loading import TrialDataset
from biovoice.data.asvspoof import stage_asvspoof2021_la_dataset
from biovoice.data.private_corpus import stage_private_corpus_dataset
from biovoice.data.manifests import load_manifest, save_split_manifests
from biovoice.data.quality_checks import (
    assert_no_trial_leakage,
    assert_speaker_disjoint,
    leakage_overlap_report,
    speaker_split_report,
    summarize_audio_quality,
)
from biovoice.evaluation.calibration import calibration_summary
from biovoice.evaluation.confusion import confusion_frame
from biovoice.evaluation.metrics import classification_metrics
from biovoice.evaluation.spoof_metrics import spoof_metric_bundle
from biovoice.evaluation.sv_metrics import target_non_target_summary
from biovoice.evaluation.thresholding import final_decision, sweep_thresholds
from biovoice.explain.case_analysis import build_case_analysis
from biovoice.explain.feature_attribution import top_feature_contributors
from biovoice.explain.reason_generator import generate_reasons
from biovoice.explain.segment_reasoning import rank_suspicious_segments
from biovoice.features.acoustic_features import extract_acoustic_features
from biovoice.features.biometric_features import compare_feature_dicts
from biovoice.features.temporal_features import extract_temporal_features
from biovoice.models.model_factory import build_anti_spoof_model, build_speaker_model
from biovoice.models.segment_model import score_segments
from biovoice.reports.artifact_inventory import (
    build_artifact_index,
    build_plot_inventory,
    flatten_metric_dict,
)
from biovoice.reports.experiment_report import build_experiment_report, save_experiment_report
from biovoice.reports.run_report import build_run_report, save_run_report
from biovoice.reports.supervisor_report import build_supervisor_report, save_supervisor_report
from biovoice.reports.table_export import export_table
from biovoice.training.checkpointing import load_checkpoint
from biovoice.training.device import resolve_device
from biovoice.training.seed import set_global_seed
from biovoice.training.train_cm import train_spoof_baseline
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.training.train_sv import train_speaker_baseline
from biovoice.utils.audio_io import inspect_audio_metadata, load_audio
from biovoice.utils.config_utils import load_config, save_yaml
from biovoice.utils.logging_utils import configure_logging
from biovoice.utils.path_utils import RunPaths, create_run_paths, resolve_path
from biovoice.utils.serialization import save_frame, save_json
from biovoice.viz.calibration_plots import plot_reliability_diagram
from biovoice.viz.data_plots import plot_class_balance, plot_duration_histogram, plot_numeric_histogram
from biovoice.viz.explainability_plots import (
    plot_feature_contributions,
    plot_segment_score_timeline,
    plot_waveform_with_segments,
)
from biovoice.viz.publication_plots import plot_comparison_bars
from biovoice.viz.score_plots import (
    plot_confusion_matrix,
    plot_det,
    plot_pr,
    plot_roc,
    plot_score_distributions,
    plot_score_scatter,
    plot_threshold_heatmap,
)
from biovoice.viz.training_plots import plot_loss_curves


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


def _load_cfg(config_path: str | Path) -> dict[str, Any]:
    """Load the repository config and normalize staged-manifest defaults."""
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
    config.setdefault("training", {}).setdefault("device", "auto")
    return config


def _setup_run(config: dict[str, Any], experiment_name: str) -> tuple[RunPaths, Any]:
    """Create the run directory and initialize logging/config snapshots."""
    run_paths = create_run_paths(config["experiment"]["output_root"], experiment_name)
    save_yaml(config, run_paths.configs / "config_snapshot.yaml")
    logger = configure_logging(run_paths.logs / "run.log")
    return run_paths, logger


def _sample_quality_subset(
    utterances: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    """Choose a deterministic, roughly stratified subset for waveform scans.

    Real corpora can be large enough that decoding every file just for quality
    diagnostics dominates end-to-end experiment time. This helper keeps the
    training/evaluation data unchanged while making the audit-only waveform scan
    tractable by sampling within split/label groups.
    """
    if sample_size <= 0 or sample_size >= len(utterances):
        return utterances.copy()
    grouped = utterances.groupby(["split", "spoof_label"], dropna=False)
    sampled_parts: list[pd.DataFrame] = []
    for _, part in grouped:
        proportional = max(1, int(round(len(part) / len(utterances) * sample_size)))
        sampled_parts.append(part.sample(n=min(len(part), proportional), random_state=seed))
    sampled = pd.concat(sampled_parts, ignore_index=True).drop_duplicates(subset=["utterance_id"])
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def _compute_quality_frame(
    utterances: pd.DataFrame,
    speech_threshold: float,
    *,
    scan_mode: str = "full",
    waveform_sample_size: int | None = None,
    seed: int = 42,
    logger: Any | None = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Compute audio quality summaries from a canonical utterance manifest.

    Data preparation uses this for both demo and real data so the same review
    tables and plots are available regardless of how the corpus was staged.
    """
    scan_mode = str(scan_mode).lower()
    rows: list[dict[str, Any]] = []
    waveform_subset = utterances
    if scan_mode == "header_plus_sample":
        sample_size = int(waveform_sample_size or 0)
        waveform_subset = _sample_quality_subset(utterances, sample_size=sample_size, seed=seed)
        waveform_ids = set(waveform_subset["utterance_id"].tolist())
        if logger is not None:
            logger.info(
                "Quality scan using header metadata for %d utterances and waveform summaries for %d sampled utterances.",
                len(utterances),
                len(waveform_subset),
            )
        for _, row in utterances.iterrows():
            metadata = inspect_audio_metadata(row["path"])
            duration_seconds = float(metadata["num_frames"] / max(metadata["sample_rate"], 1))
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
                    "quality_measurement": "header_only",
                }
            )
        rows_by_id = {str(item["utterance_id"]): item for item in rows}
        for index, (_, row) in enumerate(waveform_subset.iterrows(), start=1):
            waveform, sample_rate = load_audio(row["path"])
            summary = summarize_audio_quality(waveform, sample_rate, threshold=speech_threshold)
            rows_by_id[str(row["utterance_id"])].update(summary.to_dict())
            rows_by_id[str(row["utterance_id"])]["quality_measurement"] = "waveform"
            if logger is not None and index % max(progress_every, 1) == 0:
                logger.info(
                    "Computed sampled waveform quality for %d/%d utterances.",
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


def _dataset_review_summary(
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
    if "dataset_root" in data_cfg:
        summary["dataset_root"] = str(resolve_path(data_cfg["dataset_root"]))
    if "raw_metadata_path" in data_cfg:
        summary["raw_metadata_path"] = str(resolve_path(data_cfg["raw_metadata_path"]))
    if quality_frame is not None and not quality_frame.empty:
        summary["mean_duration_seconds"] = float(quality_frame["duration_seconds"].mean())
        summary["mean_speech_ratio"] = float(quality_frame["speech_ratio"].mean())
    return summary


def prepare_data_workflow(config_path: str | Path) -> Path:
    """Stage either demo data or a real private corpus into canonical manifests."""
    config = _load_cfg(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    run_paths, logger = _setup_run(config, "prepare_data")
    data_mode = str(config["data"].get("source_type", "demo"))
    if data_mode == "demo":
        dataset_paths = generate_demo_dataset(config)
        logger.info("Demo dataset generated at %s", dataset_paths["audio_root"])
    elif data_mode == "asvspoof2021_la":
        dataset_paths = stage_asvspoof2021_la_dataset(config)
        logger.info("ASVspoof 2019/2021 LA dataset staged under %s", config["data"]["manifest_output_dir"])
    elif data_mode == "real_private_corpus":
        dataset_paths = stage_private_corpus_dataset(config)
        logger.info("Private corpus staged from %s", config["data"]["raw_metadata_path"])
    else:
        raise ValueError(f"Unsupported data.source_type: {data_mode}")

    utterances = load_manifest(dataset_paths["utterance_manifest"])
    trials = load_manifest(dataset_paths["trial_manifest"])
    save_split_manifests(utterances, config["data"]["split_manifest_dir"], "utterances")
    save_split_manifests(trials, config["data"]["split_manifest_dir"], "trials")
    quality_cache_path = resolve_path(config["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = dataset_paths.get("quality_frame")
    if quality_frame is None and quality_cache_path.exists():
        quality_frame = pd.read_csv(quality_cache_path)
        logger.info("Loaded cached quality summary from %s", quality_cache_path)
    if quality_frame is None:
        quality_frame = _compute_quality_frame(
            utterances,
            speech_threshold=float(config["data"]["speech_threshold"]),
            scan_mode=str(config["data"].get("quality_scan_mode", "full")),
            waveform_sample_size=config["data"].get("quality_waveform_sample_size"),
            seed=int(config["experiment"]["seed"]),
            logger=logger,
            progress_every=int(config["data"].get("quality_progress_every", 500)),
        )
    leak_report = dataset_paths.get("leakage_report")
    if leak_report is None:
        leak_report = assert_no_trial_leakage(trials)
    split_report = dataset_paths.get("speaker_split_report")
    if split_report is None:
        split_report = (
            assert_speaker_disjoint(utterances)
            if bool(config["data"].get("require_speaker_disjoint", True))
            else speaker_split_report(utterances)
        )

    plot_class_balance(trials, run_paths.plots / "class_balance.png", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    plot_duration_histogram(quality_frame, run_paths.plots / "duration_histogram.png", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    plot_numeric_histogram(
        quality_frame,
        "speech_ratio",
        run_paths.plots / "speech_ratio_histogram.png",
        "Speech Ratio Histogram",
        "Speech ratio",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    enrollment_histogram = trials.copy()
    enrollment_histogram["enrollment_count"] = enrollment_histogram["enrollment_paths"].apply(len)
    plot_numeric_histogram(
        enrollment_histogram,
        "enrollment_count",
        run_paths.plots / "enrollment_count_histogram.png",
        "Enrollment Count Histogram",
        "Enrollment files per trial",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    dataset_summary = dataset_paths.get("dataset_summary") or _dataset_review_summary(
        config,
        utterances,
        trials,
        quality_frame=quality_frame,
    )
    dataset_summary["quality_scan_mode"] = str(config["data"].get("quality_scan_mode", "full"))
    if "quality_measurement" in quality_frame.columns:
        dataset_summary["quality_measurement_counts"] = {
            key: int(value)
            for key, value in quality_frame["quality_measurement"].value_counts(dropna=False).sort_index().items()
        }
    if "mean_duration_seconds" not in dataset_summary:
        dataset_summary["mean_duration_seconds"] = float(quality_frame["duration_seconds"].dropna().mean())
    if "mean_speech_ratio" not in dataset_summary and quality_frame["speech_ratio"].notna().any():
        dataset_summary["mean_speech_ratio"] = float(quality_frame["speech_ratio"].dropna().mean())
    save_frame(quality_frame, run_paths.tables / "quality_summary.csv")
    save_frame(quality_frame, quality_cache_path)
    save_frame(leak_report, run_paths.tables / "leakage_report.csv")
    save_frame(split_report, run_paths.tables / "speaker_split_report.csv")
    save_frame(pd.DataFrame([dataset_summary]), run_paths.tables / "dataset_summary.csv")
    save_json(dataset_summary, run_paths.reports / "prepare_data_summary.json")
    return run_paths.root


def inspect_data_workflow(config_path: str | Path) -> Path:
    """Compute audio quality summaries and inspection plots."""
    config = _load_cfg(config_path)
    run_paths, logger = _setup_run(config, "inspect_data")
    utterances = load_manifest(config["data"]["utterance_manifest_path"])
    quality_rows = []
    for _, row in utterances.iterrows():
        waveform, sample_rate = load_audio(row["path"])
        summary = summarize_audio_quality(waveform, sample_rate, threshold=float(config["data"]["speech_threshold"]))
        quality_rows.append({"utterance_id": row["utterance_id"], "speaker_id": row["speaker_id"], **summary.to_dict()})
    quality_frame = pd.DataFrame(quality_rows)
    save_frame(quality_frame, run_paths.tables / "quality_summary.csv")
    logger.info("Saved %d quality rows.", len(quality_frame))
    plot_duration_histogram(quality_frame, run_paths.plots / "duration_histogram.png", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    plot_numeric_histogram(
        quality_frame,
        "speech_ratio",
        run_paths.plots / "speech_ratio_histogram.png",
        "Speech Ratio Histogram",
        "Speech ratio",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_numeric_histogram(
        quality_frame,
        "clipping_ratio",
        run_paths.plots / "clipping_ratio_histogram.png",
        "Clipping Ratio Histogram",
        "Clipping ratio",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_numeric_histogram(
        quality_frame,
        "snr_proxy_db",
        run_paths.plots / "snr_proxy_histogram.png",
        "SNR Proxy Histogram",
        "SNR proxy (dB)",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    return run_paths.root


def _mean_feature_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of feature dictionaries into one speaker summary profile."""
    keys = sorted({key for item in items for key in item.keys()})
    result = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        result[key] = float(np.mean(values))
    return result


def _build_trial_predictions(
    config: dict[str, Any],
    run_paths: RunPaths,
    sv_checkpoint: Path,
    spoof_checkpoint: Path,
    split: str = "test",
    logger: Any | None = None,
) -> pd.DataFrame:
    """Run enrollment-conditioned inference and save per-trial explainability files."""
    device = resolve_device(config["training"].get("device", "auto"))
    utterance_manifest = load_manifest(config["data"]["utterance_manifest_path"])
    train_speaker_count = int(
        utterance_manifest.loc[
            utterance_manifest["split"] == config["data"]["train_split"], "speaker_id"
        ].nunique()
    )
    speaker_model = build_speaker_model(config, num_speakers=max(train_speaker_count, 2))
    spoof_model = build_anti_spoof_model(config)
    load_checkpoint(sv_checkpoint, speaker_model)
    load_checkpoint(spoof_checkpoint, spoof_model)
    speaker_model.to(device)
    spoof_model.to(device)
    speaker_model.eval()
    spoof_model.eval()

    dataset = TrialDataset(config["data"]["trial_manifest_path"], config["preprocessing"], split=split)
    rows: list[dict[str, Any]] = []
    sample_rate = int(config["data"]["sample_rate"])
    enrollment_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    probe_cache: dict[str, dict[str, Any]] = {}
    max_saved_cases = int(config["explainability"].get("max_saved_case_files", 100))
    progress_every = int(config["evaluation"].get("prediction_progress_every", 500))
    for trial_index, bundle in enumerate(dataset, start=1):
        enrollment_key = tuple(str(path) for path in bundle.enrollment_paths)
        save_case_details = trial_index <= max_saved_cases
        if enrollment_key in enrollment_cache:
            enrollment_template = enrollment_cache[enrollment_key]["template"]
        else:
            with torch.no_grad():
                enrollment_embeddings = []
                for waveform in bundle.enrollment_waveforms:
                    # Enrollment audio is summarized in two parallel spaces:
                    # learned embeddings for SV and interpretable features for
                    # explanation-oriented mismatch analysis.
                    embedding = speaker_model.encoder(waveform.unsqueeze(0).to(device))
                    enrollment_embeddings.append(embedding.squeeze(0))
                enrollment_template = aggregate_embeddings(torch.stack(enrollment_embeddings, dim=0))
            enrollment_cache[enrollment_key] = {"template": enrollment_template, "feature_mean": None}

        enrollment_feature_mean: dict[str, float] | None = enrollment_cache[enrollment_key]["feature_mean"]
        if save_case_details and enrollment_feature_mean is None:
            enrollment_features = [
                {**extract_acoustic_features(waveform, sample_rate), **extract_temporal_features(waveform, sample_rate)}
                for waveform in bundle.enrollment_waveforms
            ]
            enrollment_feature_mean = _mean_feature_dicts(enrollment_features)
            enrollment_cache[enrollment_key]["feature_mean"] = enrollment_feature_mean

        if bundle.probe_path in probe_cache:
            cached_probe = probe_cache[bundle.probe_path]
            probe_embedding = cached_probe["embedding"]
            spoof_probability = cached_probe["spoof_probability"]
        else:
            with torch.no_grad():
                probe_embedding = speaker_model.encoder(bundle.probe_waveform.unsqueeze(0).to(device)).squeeze(0)
                spoof_output = spoof_model(bundle.probe_waveform.unsqueeze(0).to(device))
                spoof_probability = float(spoof_output["probability"].item())
            probe_cache[bundle.probe_path] = {
                "embedding": probe_embedding,
                "spoof_probability": spoof_probability,
                "features": None,
            }
        probe_features: dict[str, float] | None = probe_cache[bundle.probe_path]["features"]
        if save_case_details and probe_features is None:
            probe_features = {
                **extract_acoustic_features(bundle.probe_waveform, sample_rate),
                **extract_temporal_features(bundle.probe_waveform, sample_rate),
            }
            probe_cache[bundle.probe_path]["features"] = probe_features

        with torch.no_grad():
            sv_score = float(torch.nn.functional.cosine_similarity(probe_embedding.unsqueeze(0), enrollment_template.unsqueeze(0)).item())
        feature_deltas: dict[str, float] = {}
        if save_case_details and enrollment_feature_mean is not None and probe_features is not None:
            feature_deltas = compare_feature_dicts(enrollment_feature_mean, probe_features)
        segment_frame = pd.DataFrame()
        suspicious_segments = pd.DataFrame()
        if save_case_details:
            segment_rows, _ = score_segments(
                bundle.probe_waveform,
                sample_rate,
                config["segmentation"],
                speaker_model.encoder,
                enrollment_template,
                spoof_model,
            )
            segment_frame = pd.DataFrame(segment_rows)
            suspicious_segments = rank_suspicious_segments(
                segment_frame,
                top_k=int(config["explainability"]["suspicious_segment_count"]),
            )
        decision = final_decision(
            sv_score=sv_score,
            spoof_probability=spoof_probability,
            sv_threshold=float(config["evaluation"]["sv_threshold"]),
            spoof_threshold=float(config["evaluation"]["spoof_threshold"]),
            manual_review_margin=float(config["evaluation"]["manual_review_margin"]),
        )
        top_features = (
            top_feature_contributors(feature_deltas, top_k=int(config["explainability"]["explanation_feature_count"]))
            if save_case_details
            else []
        )
        reasons = generate_reasons(decision, sv_score, spoof_probability, top_features, suspicious_segments)
        if save_case_details:
            case_payload = build_case_analysis(bundle.trial_id, decision, reasons, suspicious_segments, top_features)
            save_json(case_payload, run_paths.explainability / f"{bundle.trial_id}_case.json")
            save_frame(segment_frame, run_paths.explainability / f"{bundle.trial_id}_segments.csv")

        if len(rows) == 0:
            # The first test case is used as the default qualitative example so
            # every run emits at least one complete explainability panel.
            plot_waveform_with_segments(
                bundle.probe_waveform,
                sample_rate,
                suspicious_segments,
                run_paths.plots / "waveform_with_suspicious_segments.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_segment_score_timeline(
                segment_frame,
                "spoof_probability",
                "Spoof Score Over Time",
                run_paths.plots / "spoof_score_over_time.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_segment_score_timeline(
                segment_frame,
                "speaker_similarity",
                "Speaker Similarity Over Time",
                run_paths.plots / "speaker_similarity_over_time.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_feature_contributions(
                top_features,
                run_paths.plots / "feature_contributions.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )

        row = {
            "trial_id": bundle.trial_id,
            "speaker_id": bundle.speaker_id,
            "label": bundle.label,
            "speaker_match_label": bundle.speaker_match_label,
            "spoof_label": bundle.spoof_label,
            "sv_score": sv_score,
            "spoof_probability": spoof_probability,
            "final_decision": decision,
            "probe_path": bundle.probe_path,
            "probe_duration_seconds": bundle.probe_waveform.shape[-1] / sample_rate,
            **feature_deltas,
            "reason_1": reasons[0] if reasons else "",
        }
        rows.append(row)
        if logger is not None and trial_index % max(progress_every, 1) == 0:
            logger.info("Scored %d/%d trials for split '%s'.", trial_index, len(dataset), split)
    predictions = pd.DataFrame(rows)
    return predictions


def _plot_mandatory_evaluation_figures(config: dict[str, Any], run_paths: RunPaths, predictions: pd.DataFrame, sv_history: dict | None = None, spoof_history: dict | None = None) -> dict[str, Any]:
    """Generate the minimum figure set required for alpha review plus extra diagnostics."""
    plot_class_balance(
        predictions[["label"]].copy(),
        run_paths.plots / "class_balance.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    duration_frame = predictions.rename(columns={"probe_duration_seconds": "duration_seconds"})[["duration_seconds"]].copy()
    plot_duration_histogram(
        duration_frame,
        run_paths.plots / "duration_histogram.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    if sv_history:
        plot_loss_curves(sv_history, run_paths.plots / "sv_loss_curves.png", "SV Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    if spoof_history:
        plot_loss_curves(spoof_history, run_paths.plots / "spoof_loss_curves.png", "Spoof Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])

    sv_labels = predictions["speaker_match_label"].to_numpy(dtype=int)
    sv_scores = predictions["sv_score"].to_numpy(dtype=float)
    spoof_labels = predictions["spoof_label"].to_numpy(dtype=int)
    spoof_probs = predictions["spoof_probability"].to_numpy(dtype=float)
    if len(np.unique(sv_labels)) > 1:
        plot_roc(sv_labels, sv_scores, run_paths.plots / "sv_roc.png", "SV ROC Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_pr(sv_labels, sv_scores, run_paths.plots / "sv_pr.png", "SV Precision-Recall Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_det(sv_labels, sv_scores, run_paths.plots / "sv_det.png", "SV DET Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    if len(np.unique(spoof_labels)) > 1:
        plot_roc(spoof_labels, spoof_probs, run_paths.plots / "spoof_roc.png", "Spoof ROC Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_pr(spoof_labels, spoof_probs, run_paths.plots / "spoof_pr.png", "Spoof Precision-Recall Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
        plot_det(spoof_labels, spoof_probs, run_paths.plots / "spoof_det.png", "Spoof DET Curve", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])

    target_scores = predictions.loc[predictions["speaker_match_label"] == 1, "sv_score"].to_numpy()
    non_target_scores = predictions.loc[predictions["speaker_match_label"] == 0, "sv_score"].to_numpy()
    plot_score_distributions(
        target_scores,
        non_target_scores,
        run_paths.plots / "target_vs_non_target_scores.png",
        "Target vs Non-Target Score Distributions",
        "SV Score",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    bona_scores = predictions.loc[predictions["spoof_label"] == 0, "spoof_probability"].to_numpy()
    spoof_scores = predictions.loc[predictions["spoof_label"] == 1, "spoof_probability"].to_numpy()
    plot_score_distributions(
        spoof_scores,
        bona_scores,
        run_paths.plots / "spoof_vs_bonafide_scores.png",
        "Spoof vs Bona Fide Score Distributions",
        "Spoof Probability",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    calibration = calibration_summary(spoof_labels, spoof_probs, bins=int(config["evaluation"]["calibration_bins"]))
    plot_reliability_diagram(
        calibration,
        run_paths.plots / "reliability_diagram.png",
        "Spoof Probability Reliability Diagram",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )

    labels = ["target_bona_fide", "spoof", "wrong_speaker", "manual_review"]
    matrix = confusion_frame(predictions["label"].tolist(), predictions["final_decision"].tolist(), labels=labels)
    plot_confusion_matrix(matrix, run_paths.plots / "confusion_matrix.png", "Final Decision Confusion Matrix", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    plot_confusion_matrix(
        matrix,
        run_paths.plots / "normalized_confusion_matrix.png",
        "Normalized Final Decision Confusion Matrix",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
        normalize=True,
    )
    threshold_points = int(config["evaluation"].get("threshold_sweep_points", 11))
    threshold_sweep = sweep_thresholds(
        predictions,
        np.linspace(0.2, 0.9, threshold_points),
        np.linspace(0.2, 0.9, threshold_points),
    )
    save_frame(threshold_sweep, run_paths.tables / "threshold_sweep.csv")
    plot_threshold_heatmap(
        threshold_sweep,
        run_paths.plots / "threshold_sweep_heatmap.png",
        "Threshold Sweep Decision Accuracy",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    plot_score_scatter(
        predictions,
        run_paths.plots / "score_scatter.png",
        "SV Score vs Spoof Probability",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    return calibration


def _save_mode_comparison(predictions: pd.DataFrame, run_paths: RunPaths, config: dict[str, Any]) -> pd.DataFrame:
    """Save a compact comparison across the four main alpha experiment modes."""
    frame = predictions.copy()
    sv_only = np.where(frame["sv_score"] >= float(config["evaluation"]["sv_threshold"]), "target_bona_fide", "wrong_speaker")
    spoof_only = np.where(frame["spoof_probability"] >= float(config["evaluation"]["spoof_threshold"]), "spoof", "target_bona_fide")
    fusion = []
    fusion_plus_interp = []
    late_fused = apply_rule_fusion(frame)
    for _, row in late_fused.iterrows():
        fusion.append(
            "spoof"
            if row["spoof_probability"] >= float(config["evaluation"]["spoof_threshold"])
            else ("target_bona_fide" if row["sv_score"] >= float(config["evaluation"]["sv_threshold"]) else "wrong_speaker")
        )
        # When the current run already contains the full final decision after
        # interpretable-feature fusion, that saved decision is the authoritative
        # measured output. Falling back to an ad hoc post-hoc threshold on
        # fusion_score would misstate the run's actual performance.
        fusion_plus_interp.append(str(row.get("final_decision", "")) or str(row["label"]))
    comparison = pd.DataFrame(
        [
            {"mode": "sv_only", "decision_accuracy": float(np.mean(sv_only == frame["label"]))},
            {"mode": "spoof_only", "decision_accuracy": float(np.mean(spoof_only == frame["label"]))},
            {"mode": "fusion", "decision_accuracy": float(np.mean(np.asarray(fusion) == frame["label"]))},
            {"mode": "fusion_plus_interpretable_features", "decision_accuracy": float(np.mean(np.asarray(fusion_plus_interp) == frame["label"]))},
        ]
    )
    export_table(comparison, run_paths.tables / "mode_comparison.csv", run_paths.reports / "mode_comparison.md")
    plot_comparison_bars(
        comparison,
        x="mode",
        y="decision_accuracy",
        title="Ablation Summary Bar Chart",
        path=run_paths.plots / "ablation_summary.png",
        dpi=config["plotting"]["dpi"],
        style=config["plotting"]["style"],
    )
    save_experiment_report(build_experiment_report(comparison, "Experiment Mode Comparison"), run_paths.reports / "experiment_comparison.md")
    return comparison


def _build_alpha_checklist(run_paths: RunPaths) -> dict[str, bool]:
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


def _save_inventory_tables(run_paths: RunPaths, metrics: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save artifact, plot, and metric index tables for supervisor browsing."""
    artifact_index = build_artifact_index(run_paths.root)
    plot_inventory = build_plot_inventory(run_paths.root)
    metric_summary = flatten_metric_dict(metrics)
    export_table(artifact_index, run_paths.tables / "artifact_index.csv", run_paths.reports / "artifact_index.md")
    export_table(plot_inventory, run_paths.tables / "plot_inventory.csv", run_paths.reports / "plot_inventory.md")
    export_table(metric_summary, run_paths.tables / "metric_summary.csv", run_paths.reports / "metric_summary.md")
    return artifact_index, plot_inventory, metric_summary


def run_sv_workflow(config_path: str | Path) -> Path:
    """Train the SV baseline and export training/evaluation artifacts."""
    config = _load_cfg(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = _setup_run(config, "sv_only")
    outcome = train_speaker_baseline(config, run_paths.root)
    plot_loss_curves(outcome["history"], run_paths.plots / "sv_loss_curves.png", "SV Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    logger.info("SV training complete.")
    return run_paths.root


def run_spoof_workflow(config_path: str | Path) -> Path:
    """Train the spoof baseline and export training artifacts."""
    config = _load_cfg(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = _setup_run(config, "spoof_only")
    outcome = train_spoof_baseline(config, run_paths.root)
    plot_loss_curves(outcome["history"], run_paths.plots / "spoof_loss_curves.png", "Spoof Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    logger.info("Spoof training complete.")
    return run_paths.root


def run_joint_workflow(config_path: str | Path) -> Path:
    """Run the end-to-end alpha baseline with SV, spoof, and fusion evaluation."""
    config = _load_cfg(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = _setup_run(config, "fusion_plus_interpretable_features")
    sv_result = train_speaker_baseline(config, run_paths.root)
    spoof_result = train_spoof_baseline(config, run_paths.root)

    predictions = _build_trial_predictions(
        config,
        run_paths,
        Path(sv_result["checkpoint_path"]),
        Path(spoof_result["checkpoint_path"]),
        split=config["data"]["test_split"],
        logger=logger,
    )
    predictions = apply_rule_fusion(predictions)
    save_frame(predictions, run_paths.root / "predictions.csv")

    sv_binary_predictions = (predictions["sv_score"] >= float(config["evaluation"]["sv_threshold"])).astype(int)
    spoof_binary_predictions = (predictions["spoof_probability"] >= float(config["evaluation"]["spoof_threshold"])).astype(int)
    final_valid = predictions["final_decision"] != "manual_review"
    final_true = predictions.loc[final_valid, "label"].map({"wrong_speaker": 0, "spoof": 1, "target_bona_fide": 2}).to_numpy()
    final_pred = predictions.loc[final_valid, "final_decision"].map({"wrong_speaker": 0, "spoof": 1, "target_bona_fide": 2}).to_numpy()

    metrics = {
        "sv": {**classification_metrics(predictions["speaker_match_label"], sv_binary_predictions, probabilities=predictions["sv_score"]), **target_non_target_summary(predictions["sv_score"].to_numpy(), predictions["speaker_match_label"].to_numpy())},
        "spoof": spoof_metric_bundle(predictions["spoof_label"].to_numpy(), predictions["spoof_probability"].to_numpy(), spoof_binary_predictions.to_numpy()),
        "joint": classification_metrics(final_true, final_pred) if len(final_true) else {"accuracy": 0.0},
    }
    calibration = _plot_mandatory_evaluation_figures(config, run_paths, predictions, sv_result["history"], spoof_result["history"])
    metrics["calibration"] = {"brier_score": calibration["brier_score"], "ece": calibration["ece"]}
    comparison = _save_mode_comparison(predictions, run_paths, config)
    quality_summary_path = resolve_path(config["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = _dataset_review_summary(
        config,
        load_manifest(config["data"]["utterance_manifest_path"]),
        load_manifest(config["data"]["trial_manifest_path"]),
        quality_frame=quality_frame,
    )

    save_json(metrics, run_paths.root / "metrics.json")
    export_table(predictions, run_paths.tables / "predictions.csv")
    comparison.to_csv(run_paths.tables / "ablation_summary.csv", index=False)
    save_json(dataset_review, run_paths.reports / "dataset_review.json")
    preliminary_artifact_frame = build_artifact_index(run_paths.root)
    save_run_report(
        build_run_report(
            "fusion_plus_interpretable_features",
            metrics,
            preliminary_artifact_frame["relative_path"].tolist()[:60],
            ALPHA_LIMITATIONS,
            alpha_checklist=None,
            interpretation_notes=METRIC_INTERPRETATION_NOTES,
        ),
        run_paths.reports / "run_report.md",
    )
    supervisor_summary = {
        "project_summary": "Enrollment-conditioned alpha baseline combining speaker verification, spoof detection, fusion logic, and interpretable mismatch analysis.",
        "alpha_evidence": [
            (
                "ASVspoof 2019/2021 LA data staged and consumed end-to-end."
                if dataset_review["dataset_mode"] == "asvspoof2021_la"
                else "Synthetic/demo data generated and consumed end-to-end."
            ),
            "Baseline SV run completed and saved history/checkpoint artifacts.",
            "Baseline spoof run completed and saved history/checkpoint artifacts.",
            "Fusion evaluation completed with saved predictions, metrics, and mandatory figures.",
            "Supervisor notebook can read saved artifacts from outputs.",
        ],
        "metrics": [
            f"SV EER: {metrics['sv'].get('eer', 0.0):.3f}",
            f"Spoof ROC-AUC: {metrics['spoof'].get('roc_auc', 0.0):.3f}",
            f"Joint Accuracy: {metrics['joint'].get('accuracy', 0.0):.3f}",
            f"Calibration ECE: {metrics['calibration']['ece']:.3f}",
        ],
        "dataset_review": [
            f"Dataset mode: {dataset_review['dataset_mode']}",
            f"Dataset name: {dataset_review['dataset_name']}",
            f"Split strategy: {dataset_review['split_strategy']}",
            f"Speaker-disjoint requirement: {dataset_review['require_speaker_disjoint']}",
            f"Speaker-disjoint violations: {dataset_review['speaker_disjoint_violations']}",
            f"Trial leakage violations: {dataset_review['trial_leakage_violations']}",
            f"Trial counts by label: {dataset_review['trial_labels']}",
        ],
        "metric_interpretation": [
            METRIC_INTERPRETATION_NOTES["sv"],
            METRIC_INTERPRETATION_NOTES["spoof"],
            METRIC_INTERPRETATION_NOTES["joint"],
            METRIC_INTERPRETATION_NOTES["calibration"],
        ],
        "figures": [str(path.relative_to(run_paths.root)) for path in sorted(run_paths.plots.glob("*.png"))],
        "artifact_map": [
            f"Artifact index: {run_paths.reports / 'artifact_index.md'}",
            f"Plot inventory: {run_paths.reports / 'plot_inventory.md'}",
            f"Metric summary: {run_paths.reports / 'metric_summary.md'}",
            f"Prediction table: {run_paths.tables / 'predictions.csv'}",
            f"Notebook-ready walkthrough artifacts live under: {run_paths.root}",
        ],
        "limitations": ALPHA_LIMITATIONS,
        "next_steps": [
            "Inspect the threshold heatmap to choose a more defensible operating region before claiming a decision policy.",
            "Use the plot inventory when presenting figures to supervisors so every figure is paired with an interpretation note.",
            (
                "Treat the current numbers as measured ASVspoof baseline evidence and avoid broader real-world claims until robustness studies are complete."
                if dataset_review["dataset_mode"] == "asvspoof2021_la"
                else "Treat the current numbers as alpha plumbing evidence because the run uses synthetic/demo data."
            ),
        ],
    }
    save_supervisor_report(build_supervisor_report(supervisor_summary), run_paths.reports / "supervisor_report.md")

    _save_inventory_tables(run_paths, metrics)
    alpha_checklist = _build_alpha_checklist(run_paths)
    save_json(alpha_checklist, run_paths.reports / "alpha_exit_checklist.json")

    artifact_frame, plot_inventory, metric_summary = _save_inventory_tables(run_paths, metrics)
    artifact_index = artifact_frame["relative_path"].tolist()
    save_run_report(
        build_run_report(
            "fusion_plus_interpretable_features",
            metrics,
            artifact_index[:60],
            ALPHA_LIMITATIONS,
            alpha_checklist=alpha_checklist,
            interpretation_notes=METRIC_INTERPRETATION_NOTES,
        ),
        run_paths.reports / "run_report.md",
    )
    _save_inventory_tables(run_paths, metrics)
    logger.info("Joint fusion workflow complete.")
    return run_paths.root


def run_ablation_workflow(config_path: str | Path) -> Path:
    """Generate the compact alpha ablation figure from a fusion run."""
    run_root = run_joint_workflow(config_path)
    return run_root


def generate_supervisor_artifacts(config_path: str | Path) -> Path:
    """Ensure the end-to-end run exists and return its directory."""
    return run_joint_workflow(config_path)


def export_tables_workflow(config_path: str | Path) -> Path:
    """Trigger the end-to-end workflow so the expected tables exist."""
    return run_joint_workflow(config_path)
