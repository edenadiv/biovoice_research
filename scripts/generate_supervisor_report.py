"""Generate supervisor-facing artifacts."""

from __future__ import annotations

import typer

from biovoice.workflows import generate_supervisor_artifacts


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(generate_supervisor_artifacts(config)))


if __name__ == "__main__":
    typer.run(main)
