"""Resume final fusion scoring from existing checkpoints.

This script is intentionally narrow: it lets us recover a partially completed
joint run after training has finished but trial scoring or report generation
failed. That is especially useful for long real-data baselines where rerunning
training would waste substantial wall-clock time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from biovoice.data.manifests import load_manifest
from biovoice.utils.logging_utils import configure_logging
from biovoice.utils.path_utils import RunPaths, resolve_path
from biovoice.workflows.common import load_workflow_config, merge_dataset_review
from biovoice.workflows.evaluation import evaluate_joint_predictions, prepare_threshold_selection
from biovoice.workflows.inference import build_trial_predictions
from biovoice.workflows.reporting import write_joint_run_outputs

app = typer.Typer(add_completion=False)


def _run_paths_from_root(run_root: Path) -> RunPaths:
    """Rebuild the standard run-path view for an existing run directory."""
    return RunPaths(
        root=run_root,
        logs=run_root / "logs",
        configs=run_root / "configs",
        checkpoints=run_root / "checkpoints",
        plots=run_root / "plots",
        tables=run_root / "tables",
        reports=run_root / "reports",
        explainability=run_root / "explainability",
        calibration=run_root / "calibration",
        ablations=run_root / "ablations",
    )


@app.command()
def main(
    config: str = "configs/default.yaml",
    run_dir: str = "outputs/runs",
    reuse_saved_test_predictions: bool = True,
) -> None:
    """Resume a partial joint run using its saved checkpoints and histories."""
    config_dict = load_workflow_config(config)
    run_root = resolve_path(run_dir)
    run_paths = _run_paths_from_root(run_root)
    logger = configure_logging(run_paths.logs / "run.log")
    logger.info("Resuming final scoring from existing checkpoints in %s", run_root)

    sv_checkpoint = run_paths.checkpoints / "speaker_model.pt"
    spoof_checkpoint = run_paths.checkpoints / "spoof_model.pt"
    if not sv_checkpoint.exists() or not spoof_checkpoint.exists():
        raise FileNotFoundError(
            "Expected both speaker_model.pt and spoof_model.pt before resuming scoring."
        )

    with open(run_paths.reports / "sv_history.json", "r", encoding="utf-8") as handle:
        sv_history = json.load(handle)
    with open(run_paths.reports / "spoof_history.json", "r", encoding="utf-8") as handle:
        spoof_history = json.load(handle)

    validation_predictions = build_trial_predictions(
        config_dict,
        run_paths,
        sv_checkpoint,
        spoof_checkpoint,
        split=config_dict["data"]["validation_split"],
        logger=logger,
    )
    if validation_predictions.empty:
        logger.warning(
            "Validation trial split '%s' is empty; falling back to test predictions for threshold tuning. "
            "This fallback is for smoke/demo recovery only, not for real-data claims.",
            config_dict["data"]["validation_split"],
        )
        validation_predictions = build_trial_predictions(
            config_dict,
            run_paths,
            sv_checkpoint,
            spoof_checkpoint,
            split=config_dict["data"]["test_split"],
            logger=logger,
        )
    threshold_sweep, selected_thresholds = prepare_threshold_selection(config_dict, validation_predictions)
    use_tuned_thresholds = bool(config_dict["evaluation"].get("use_tuned_thresholds", True))
    active_sv_threshold = float(selected_thresholds["sv_threshold"] if use_tuned_thresholds else config_dict["evaluation"]["sv_threshold"])
    active_spoof_threshold = float(selected_thresholds["spoof_threshold"] if use_tuned_thresholds else config_dict["evaluation"]["spoof_threshold"])
    saved_predictions_path = run_paths.root / "predictions.csv"
    if reuse_saved_test_predictions and saved_predictions_path.exists():
        predictions = pd.read_csv(saved_predictions_path)
        logger.info("Reused saved test predictions from %s", saved_predictions_path)
    else:
        predictions = build_trial_predictions(
            config_dict,
            run_paths,
            sv_checkpoint,
            spoof_checkpoint,
            split=config_dict["data"]["test_split"],
            sv_threshold=active_sv_threshold,
            spoof_threshold=active_spoof_threshold,
            logger=logger,
        )
    predictions, metrics, comparison, analysis = evaluate_joint_predictions(
        config_dict,
        run_paths,
        predictions,
        validation_predictions=validation_predictions,
        threshold_sweep=threshold_sweep,
        selected_thresholds=selected_thresholds,
        sv_history=sv_history,
        spoof_history=spoof_history,
    )

    quality_summary_path = resolve_path(config_dict["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = merge_dataset_review(
        config_dict,
        load_manifest(config_dict["data"]["utterance_manifest_path"]),
        load_manifest(config_dict["data"]["trial_manifest_path"]),
        quality_frame=quality_frame,
    )

    write_joint_run_outputs(run_paths, metrics, predictions, comparison, dataset_review, analysis=analysis)
    logger.info("Completed resumed scoring and reporting for %s", run_root)
    typer.echo(str(run_root))


if __name__ == "__main__":
    app()
