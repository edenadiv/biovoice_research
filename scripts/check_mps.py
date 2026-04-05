"""Small helper for validating Apple Metal / MPS availability."""

from __future__ import annotations

import platform

import torch
import typer

from biovoice.training.device import resolve_device


app = typer.Typer(add_completion=False)


@app.command()
def main(requested: str = "auto") -> None:
    """Print backend availability and run a tiny tensor smoke test."""
    typer.echo(f"platform={platform.platform()}")
    typer.echo(f"machine={platform.machine()}")
    typer.echo(f"torch={torch.__version__}")
    typer.echo(f"mps_built={torch.backends.mps.is_built()}")
    typer.echo(f"mps_available={torch.backends.mps.is_available()}")
    resolved = resolve_device(requested)
    typer.echo(f"resolved_device={resolved}")
    tensor = torch.ones(4, device=resolved)
    typer.echo(f"tensor_device={tensor.device}")
    typer.echo(f"tensor_sum={float(tensor.sum().item())}")


if __name__ == "__main__":
    app()
