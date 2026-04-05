"""Train the baseline anti-spoof branch."""

from __future__ import annotations

import typer

from biovoice.workflows import run_spoof_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(run_spoof_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
