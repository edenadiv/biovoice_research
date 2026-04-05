"""Generate the synthetic/demo dataset used for alpha smoke tests."""

from __future__ import annotations

import typer

from biovoice.workflows import prepare_data_workflow


def main(config: str = "configs/default.yaml") -> None:
    typer.echo(str(prepare_data_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
