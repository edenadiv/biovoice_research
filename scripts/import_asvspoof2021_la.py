"""Stage the official ASVspoof 2019/2021 LA data into canonical manifests."""

from __future__ import annotations

import typer

from biovoice.workflows import prepare_data_workflow


def main(config: str = "configs/asvspoof2021_la.yaml") -> None:
    typer.echo(str(prepare_data_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
