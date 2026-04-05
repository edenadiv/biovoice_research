"""Inspect data quality and preprocessing statistics."""

from __future__ import annotations

import typer

from biovoice.workflows import inspect_data_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(inspect_data_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
