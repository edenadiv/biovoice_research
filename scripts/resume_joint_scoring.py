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
from biovoice.evaluation.metrics import classification_metrics
from biovoice.evaluation.spoof_metrics import spoof_metric_bundle
from biovoice.evaluation.sv_metrics import target_non_target_summary
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.utils.config_utils import load_config
from biovoice.utils.logging_utils import configure_logging
from biovoice.utils.path_utils import RunPaths, resolve_path
from biovoice.utils.serialization import save_frame
from biovoice.workflows.common import merge_dataset_review
from biovoice.workflows.evaluation import plot_mandatory_evaluation_figures, save_mode_comparison
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
) -> None:
    """Resume a partial joint run using its saved checkpoints and histories."""
    config_dict = load_config(config)
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

    predictions = build_trial_predictions(
        config_dict,
        run_paths,
        sv_checkpoint,
        spoof_checkpoint,
        split=config_dict["data"]["test_split"],
        logger=logger,
    )
    predictions = apply_rule_fusion(predictions)
    save_frame(predictions, run_paths.root / "predictions.csv")

    sv_binary_predictions = (
        predictions["sv_score"] >= float(config_dict["evaluation"]["sv_threshold"])
    ).astype(int)
    spoof_binary_predictions = (
        predictions["spoof_probability"] >= float(config_dict["evaluation"]["spoof_threshold"])
    ).astype(int)
    final_valid = predictions["final_decision"] != "manual_review"
    final_true = predictions.loc[final_valid, "label"].map(
        {"wrong_speaker": 0, "spoof": 1, "target_bona_fide": 2}
    ).to_numpy()
    final_pred = predictions.loc[final_valid, "final_decision"].map(
        {"wrong_speaker": 0, "spoof": 1, "target_bona_fide": 2}
    ).to_numpy()

    metrics = {
        "sv": {
            **classification_metrics(
                predictions["speaker_match_label"],
                sv_binary_predictions,
                probabilities=predictions["sv_score"],
            ),
            **target_non_target_summary(
                predictions["sv_score"].to_numpy(),
                predictions["speaker_match_label"].to_numpy(),
            ),
        },
        "spoof": spoof_metric_bundle(
            predictions["spoof_label"].to_numpy(),
            predictions["spoof_probability"].to_numpy(),
            spoof_binary_predictions.to_numpy(),
        ),
        "joint": classification_metrics(final_true, final_pred)
        if len(final_true)
        else {"accuracy": 0.0},
    }
    calibration = plot_mandatory_evaluation_figures(
        config_dict,
        run_paths,
        predictions,
        sv_history,
        spoof_history,
    )
    metrics["calibration"] = {
        "brier_score": calibration["brier_score"],
        "ece": calibration["ece"],
    }
    comparison = save_mode_comparison(predictions, run_paths, config_dict)

    quality_summary_path = resolve_path(config_dict["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = merge_dataset_review(
        config_dict,
        load_manifest(config_dict["data"]["utterance_manifest_path"]),
        load_manifest(config_dict["data"]["trial_manifest_path"]),
        quality_frame=quality_frame,
    )

    write_joint_run_outputs(run_paths, metrics, predictions, comparison, dataset_review)
    logger.info("Completed resumed scoring and reporting for %s", run_root)
    typer.echo(str(run_root))


if __name__ == "__main__":
    app()
