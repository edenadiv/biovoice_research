"""Run the end-to-end fusion workflow."""

from __future__ import annotations

import typer

from biovoice.workflows import run_joint_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_joint_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
