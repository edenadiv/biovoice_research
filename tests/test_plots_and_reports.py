"""Tests for plot generation and supervisor-facing report helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biovoice.reports.artifact_inventory import build_artifact_index, build_plot_inventory
from biovoice.reports.run_report import build_run_report
from biovoice.reports.supervisor_report import build_supervisor_report
from biovoice.viz.score_plots import (
    plot_confusion_matrix,
    plot_score_scatter,
    plot_threshold_heatmap,
)


def test_score_plot_utilities_create_files(tmp_path: Path) -> None:
    confusion = pd.DataFrame(
        [[4, 1], [2, 3]],
        index=["target_bona_fide", "spoof"],
        columns=["target_bona_fide", "spoof"],
    )
    thresholds = pd.DataFrame(
        [
            {"sv_threshold": 0.3, "spoof_threshold": 0.3, "decision_accuracy": 0.4},
            {"sv_threshold": 0.5, "spoof_threshold": 0.3, "decision_accuracy": 0.6},
            {"sv_threshold": 0.3, "spoof_threshold": 0.5, "decision_accuracy": 0.5},
            {"sv_threshold": 0.5, "spoof_threshold": 0.5, "decision_accuracy": 0.7},
        ]
    )
    scatter_frame = pd.DataFrame(
        {
            "sv_score": np.asarray([0.2, 0.8, 0.9]),
            "spoof_probability": np.asarray([0.9, 0.3, 0.2]),
            "label": ["spoof", "wrong_speaker", "target_bona_fide"],
        }
    )
    plot_confusion_matrix(confusion, tmp_path / "normalized_confusion.png", "Normalized", normalize=True)
    plot_threshold_heatmap(thresholds, tmp_path / "threshold_heatmap.png", "Threshold Sweep")
    plot_score_scatter(scatter_frame, tmp_path / "score_scatter.png", "Joint Score Scatter")
    assert (tmp_path / "normalized_confusion.png").exists()
    assert (tmp_path / "threshold_heatmap.png").exists()
    assert (tmp_path / "score_scatter.png").exists()


def test_artifact_inventory_and_plot_notes(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    plots = run_root / "plots"
    tables = run_root / "tables"
    plots.mkdir(parents=True)
    tables.mkdir(parents=True)
    (plots / "class_balance.png").write_bytes(b"png")
    (tables / "metrics.csv").write_text("metric,value\naccuracy,1.0\n", encoding="utf-8")
    artifact_index = build_artifact_index(run_root)
    plot_inventory = build_plot_inventory(run_root)
    assert "relative_path" in artifact_index.columns
    assert "what_it_shows" in plot_inventory.columns
    assert plot_inventory.iloc[0]["filename"] == "class_balance.png"


def test_reports_include_supervisor_sections() -> None:
    run_report = build_run_report(
        "demo",
        {"joint": {"accuracy": 0.5}},
        ["plots/confusion_matrix.png"],
        ["Synthetic-only evidence."],
        alpha_checklist={"Mandatory plots were generated": True},
        interpretation_notes={"joint": "Use this as the headline metric."},
    )
    supervisor_report = build_supervisor_report(
        {
            "project_summary": "Summary",
            "alpha_evidence": ["Evidence"],
            "metrics": ["Accuracy: 0.5"],
            "dataset_review": ["Dataset mode: real_private_corpus"],
            "metric_interpretation": ["Interpret carefully."],
            "figures": ["plots/confusion_matrix.png"],
            "artifact_map": ["reports/artifact_index.md"],
            "limitations": ["Synthetic data only."],
            "next_steps": ["Run a real dataset benchmark."],
        }
    )
    assert "Alpha Exit Checklist" in run_report
    assert "Interpretation Notes" in run_report
    assert "How To Read This Report" in supervisor_report
    assert "Dataset Review" in supervisor_report
    assert "Artifact Map" in supervisor_report
