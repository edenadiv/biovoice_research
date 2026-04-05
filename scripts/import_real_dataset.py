"""Convenience entrypoint for real-data staging.

This script is intentionally thin: it reuses the standard prepare-data
workflow so the staged manifests and review artifacts stay consistent with the
rest of the repository.
"""

from __future__ import annotations

import typer

from biovoice.workflows import prepare_data_workflow


def main(config: str = "configs/private_corpus_template.yaml") -> None:
    typer.echo(str(prepare_data_workflow(config)))


if __name__ == "__main__":
    typer.run(main)
