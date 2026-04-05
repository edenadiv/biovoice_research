"""Export result tables by running the end-to-end workflow."""

from __future__ import annotations

import typer

from biovoice.workflows import export_tables_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(export_tables_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
