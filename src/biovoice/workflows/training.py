"""Top-level training and end-to-end experiment workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from biovoice.data.manifests import load_manifest
from biovoice.evaluation.metrics import classification_metrics
from biovoice.evaluation.spoof_metrics import spoof_metric_bundle
from biovoice.evaluation.sv_metrics import target_non_target_summary
from biovoice.training.seed import set_global_seed
from biovoice.training.train_cm import train_spoof_baseline
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.training.train_sv import train_speaker_baseline
from biovoice.utils.path_utils import resolve_path
from biovoice.viz.training_plots import plot_loss_curves

from .common import load_workflow_config, merge_dataset_review, setup_run
from .data_prep import prepare_data_workflow
from .evaluation import plot_mandatory_evaluation_figures, save_mode_comparison
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

    predictions = build_trial_predictions(
        config,
        run_paths,
        Path(sv_result["checkpoint_path"]),
        Path(spoof_result["checkpoint_path"]),
        split=config["data"]["test_split"],
        logger=logger,
    )
    predictions = apply_rule_fusion(predictions)
    from biovoice.utils.serialization import save_frame

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
    calibration = plot_mandatory_evaluation_figures(config, run_paths, predictions, sv_result["history"], spoof_result["history"])
    metrics["calibration"] = {"brier_score": calibration["brier_score"], "ece": calibration["ece"]}
    comparison = save_mode_comparison(predictions, run_paths, config)

    quality_summary_path = resolve_path(config["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = merge_dataset_review(
        config,
        load_manifest(config["data"]["utterance_manifest_path"]),
        load_manifest(config["data"]["trial_manifest_path"]),
        quality_frame=quality_frame,
    )

    write_joint_run_outputs(run_paths, metrics, predictions, comparison, dataset_review)
    logger.info("Joint fusion workflow complete.")
    return run_paths.root


def run_ablation_workflow(config_path: str | Path) -> Path:
    """Generate the compact alpha ablation figure from a fusion run."""
    return run_joint_workflow(config_path)
