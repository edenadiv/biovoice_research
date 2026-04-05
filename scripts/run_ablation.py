"""Generate the compact alpha ablation summary."""

from __future__ import annotations

import typer

from biovoice.workflows import run_ablation_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_ablation_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
