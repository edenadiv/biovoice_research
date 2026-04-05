"""Typer CLI for the research repository."""

from __future__ import annotations

from pathlib import Path

import typer

from biovoice.workflows import (
    export_tables_workflow,
    generate_supervisor_artifacts,
    inspect_data_workflow,
    prepare_data_workflow,
    run_ablation_workflow,
    run_joint_workflow,
    run_spoof_workflow,
    run_sv_workflow,
)


app = typer.Typer(add_completion=False, help="Research CLI for the BioVoice alpha repository.")


@app.command("prepare-data")
def prepare_data(config: str = "configs/default.yaml") -> None:
    typer.echo(str(prepare_data_workflow(Path(config))))


@app.command("inspect-data")
def inspect_data(config: str = "configs/default.yaml") -> None:
    typer.echo(str(inspect_data_workflow(Path(config))))


@app.command("train-sv")
def train_sv(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_sv_workflow(Path(config))))


@app.command("train-spoof")
def train_spoof(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_spoof_workflow(Path(config))))


@app.command("train-joint")
def train_joint(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_joint_workflow(Path(config))))


@app.command("run-ablation")
def run_ablation(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_ablation_workflow(Path(config))))


@app.command("generate-supervisor-report")
def generate_supervisor_report(config: str = "configs/default.yaml") -> None:
    typer.echo(str(generate_supervisor_artifacts(Path(config))))


@app.command("export-tables")
def export_tables(config: str = "configs/default.yaml") -> None:
    typer.echo(str(export_tables_workflow(Path(config))))


if __name__ == "__main__":
    app()
