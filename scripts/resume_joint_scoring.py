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
from biovoice.reports.artifact_inventory import build_artifact_index
from biovoice.reports.run_report import build_run_report, save_run_report
from biovoice.reports.supervisor_report import build_supervisor_report, save_supervisor_report
from biovoice.reports.table_export import export_table
from biovoice.training.train_joint import apply_rule_fusion
from biovoice.utils.config_utils import load_config
from biovoice.utils.logging_utils import configure_logging
from biovoice.utils.path_utils import RunPaths, resolve_path
from biovoice.utils.serialization import save_frame, save_json
from biovoice.workflows import (
    ALPHA_LIMITATIONS,
    METRIC_INTERPRETATION_NOTES,
    _build_alpha_checklist,
    _build_trial_predictions,
    _dataset_review_summary,
    _plot_mandatory_evaluation_figures,
    _save_inventory_tables,
    _save_mode_comparison,
)

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

    predictions = _build_trial_predictions(
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
    calibration = _plot_mandatory_evaluation_figures(
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
    comparison = _save_mode_comparison(predictions, run_paths, config_dict)

    quality_summary_path = resolve_path(config_dict["data"]["manifest_output_dir"]) / "quality_summary.csv"
    quality_frame = pd.read_csv(quality_summary_path) if quality_summary_path.exists() else None
    dataset_review = _dataset_review_summary(
        config_dict,
        load_manifest(config_dict["data"]["utterance_manifest_path"]),
        load_manifest(config_dict["data"]["trial_manifest_path"]),
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
            "ASVspoof 2019/2021 LA data staged and consumed end-to-end.",
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
            "Treat the current numbers as measured ASVspoof baseline evidence and avoid broader real-world claims until robustness studies are complete.",
        ],
    }
    save_supervisor_report(
        build_supervisor_report(supervisor_summary),
        run_paths.reports / "supervisor_report.md",
    )

    _save_inventory_tables(run_paths, metrics)
    alpha_checklist = _build_alpha_checklist(run_paths)
    save_json(alpha_checklist, run_paths.reports / "alpha_exit_checklist.json")
    artifact_frame, _, _ = _save_inventory_tables(run_paths, metrics)
    save_run_report(
        build_run_report(
            "fusion_plus_interpretable_features",
            metrics,
            artifact_frame["relative_path"].tolist()[:60],
            ALPHA_LIMITATIONS,
            alpha_checklist=alpha_checklist,
            interpretation_notes=METRIC_INTERPRETATION_NOTES,
        ),
        run_paths.reports / "run_report.md",
    )
    _save_inventory_tables(run_paths, metrics)
    logger.info("Completed resumed scoring and reporting for %s", run_root)
    typer.echo(str(run_root))


if __name__ == "__main__":
    app()
