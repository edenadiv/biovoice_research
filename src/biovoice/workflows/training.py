"""Top-level training and end-to-end experiment workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biovoice.data.manifests import load_manifest
from biovoice.training.seed import set_global_seed
from biovoice.training.train_cm import train_spoof_baseline
from biovoice.training.train_sv import train_speaker_baseline
from biovoice.utils.path_utils import resolve_path
from biovoice.viz.training_plots import plot_loss_curves

from .common import load_workflow_config, merge_dataset_review, setup_run
from .data_prep import prepare_data_workflow
from .evaluation import evaluate_joint_predictions, prepare_threshold_selection
from .inference import build_trial_predictions
from .reporting import write_joint_run_outputs


def run_sv_workflow(config_path: str | Path) -> Path:
    """Train the SV baseline and export training/evaluation artifacts."""
    config = load_workflow_config(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = setup_run(config, "sv_only")
    outcome = train_speaker_baseline(config, run_paths.root)
    plot_loss_curves(outcome["history"], run_paths.plots / "sv_loss_curves.png", "SV Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    logger.info("SV training complete.")
    return run_paths.root


def run_spoof_workflow(config_path: str | Path) -> Path:
    """Train the spoof baseline and export training artifacts."""
    config = load_workflow_config(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = setup_run(config, "spoof_only")
    outcome = train_spoof_baseline(config, run_paths.root)
    plot_loss_curves(outcome["history"], run_paths.plots / "spoof_loss_curves.png", "Spoof Train/Validation Loss", dpi=config["plotting"]["dpi"], style=config["plotting"]["style"])
    logger.info("Spoof training complete.")
    return run_paths.root


def run_joint_workflow(config_path: str | Path) -> Path:
    """Run the end-to-end alpha baseline with SV, spoof, and fusion evaluation."""
    config = load_workflow_config(config_path)
    set_global_seed(int(config["experiment"]["seed"]))
    prepare_data_workflow(config_path)
    run_paths, logger = setup_run(config, "fusion_plus_interpretable_features")
    sv_result = train_speaker_baseline(config, run_paths.root)
    spoof_result = train_spoof_baseline(config, run_paths.root)

    validation_predictions = build_trial_predictions(
        config,
        run_paths,
        Path(sv_result["checkpoint_path"]),
        Path(spoof_result["checkpoint_path"]),
        split=config["data"]["validation_split"],
        logger=logger,
    )
    if validation_predictions.empty:
        logger.warning(
            "Validation trial split '%s' is empty; falling back to test predictions for threshold tuning. "
            "This is acceptable for tiny smoke/demo runs but should not be used for real-data claims.",
            config["data"]["validation_split"],
        )
        validation_predictions = build_trial_predictions(
            config,
            run_paths,
            Path(sv_result["checkpoint_path"]),
            Path(spoof_result["checkpoint_path"]),
            split=config["data"]["test_split"],
            logger=logger,
        )
    threshold_sweep, selected_thresholds = prepare_threshold_selection(config, validation_predictions)
    use_tuned_thresholds = bool(config["evaluation"].get("use_tuned_thresholds", True))
    active_sv_threshold = float(selected_thresholds["sv_threshold"] if use_tuned_thresholds else config["evaluation"]["sv_threshold"])
    active_spoof_threshold = float(selected_thresholds["spoof_threshold"] if use_tuned_thresholds else config["evaluation"]["spoof_threshold"])
    predictions = build_trial_predictions(
        config,
        run_paths,
        Path(sv_result["checkpoint_path"]),
        Path(spoof_result["checkpoint_path"]),
        split=config["data"]["test_split"],
        sv_threshold=active_sv_threshold,
        spoof_threshold=active_spoof_threshold,
        logger=logger,
    )
    predictions, metrics, comparison, analysis = evaluate_joint_predictions(
        config,
        run_paths,
        predictions,
        validation_predictions=validation_predictions,
        threshold_sweep=threshold_sweep,
        selected_thresholds=selected_thresholds,
        sv_history=sv_result["history"],
        spoof_history=spoof_result["history"],
    )

    quality_summary_path = resolve_path(config["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = merge_dataset_review(
        config,
        load_manifest(config["data"]["utterance_manifest_path"]),
        load_manifest(config["data"]["trial_manifest_path"]),
        quality_frame=quality_frame,
    )

    write_joint_run_outputs(run_paths, metrics, predictions, comparison, dataset_review, analysis=analysis)
    logger.info("Joint fusion workflow complete.")
    return run_paths.root


def run_ablation_workflow(config_path: str | Path) -> Path:
    """Generate the compact alpha ablation figure from a fusion run."""
    return run_joint_workflow(config_path)
