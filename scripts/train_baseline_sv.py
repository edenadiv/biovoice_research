"""Train the baseline speaker verification branch."""

from __future__ import annotations

import typer

from biovoice.workflows import run_sv_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_sv_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
