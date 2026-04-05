"""Data-preparation workflows for demo, ASVspoof, and private-corpus staging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biovoice.data.asvspoof import stage_asvspoof2021_la_dataset
from biovoice.data.demo import generate_demo_dataset
from biovoice.data.manifests import load_manifest, save_split_manifests
from biovoice.data.private_corpus import stage_private_corpus_dataset
from biovoice.data.quality_checks import assert_no_trial_leakage, assert_speaker_disjoint, speaker_split_report, summarize_audio_quality
from biovoice.utils.audio_io import load_audio
from biovoice.utils.path_utils import resolve_path
from biovoice.utils.serialization import save_frame, save_json
from biovoice.viz.data_plots import plot_class_balance, plot_duration_histogram, plot_numeric_histogram

from .common import compute_quality_frame, load_workflow_config, merge_dataset_review, setup_run


def prepare_data_workflow(config_path: str | Path) -> Path:
    """Stage demo or real data into canonical manifests plus audit artifacts."""
    config = load_workflow_config(config_path)
    from biovoice.training.seed import set_global_seed

    set_global_seed(int(config["experiment"]["seed"]))
    run_paths, logger = setup_run(config, "prepare_data")
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
        quality_frame = compute_quality_frame(
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
    if "clipping_ratio" in quality_frame.columns:
        plot_numeric_histogram(
            quality_frame,
            "clipping_ratio",
            run_paths.plots / "clipping_ratio_histogram.png",
            "Clipping Ratio Histogram",
            "Clipping ratio",
            dpi=config["plotting"]["dpi"],
            style=config["plotting"]["style"],
        )
    if "snr_proxy_db" in quality_frame.columns:
        plot_numeric_histogram(
            quality_frame,
            "snr_proxy_db",
            run_paths.plots / "snr_proxy_histogram.png",
            "SNR Proxy Histogram",
            "SNR proxy (dB)",
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

    dataset_summary = merge_dataset_review(config, utterances, trials, quality_frame=quality_frame)
    save_frame(quality_frame, run_paths.tables / "quality_summary.csv")
    save_frame(quality_frame, quality_cache_path)
    save_frame(leak_report, run_paths.tables / "leakage_report.csv")
    save_frame(split_report, run_paths.tables / "speaker_split_report.csv")
    save_frame(pd.DataFrame([dataset_summary]), run_paths.tables / "dataset_summary.csv")
    save_json(dataset_summary, run_paths.reports / "prepare_data_summary.json")
    return run_paths.root


def inspect_data_workflow(config_path: str | Path) -> Path:
    """Compute stand-alone quality summaries and plots from staged utterances."""
    config = load_workflow_config(config_path)
    run_paths, logger = setup_run(config, "inspect_data")
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
