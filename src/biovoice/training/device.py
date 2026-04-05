"""Device-selection helpers for CPU, CUDA, and Apple MPS backends.

The repository started CPU-first for reproducibility, but larger real-data
experiments benefit substantially from explicit accelerator support. This
module centralizes device resolution so training, scoring, and smoke tests all
follow the same policy.
"""

from __future__ import annotations

import torch


def resolve_device(requested: str | None) -> str:
    """Resolve a user/device-config request into an actual torch device string.

    Supported requests:
    - ``"auto"``: prefer CUDA, then Apple MPS, then CPU
    - ``"cuda"`` / ``"mps"`` / ``"cpu"``: require that device or raise
    - ``None`` / empty: behave like ``"auto"``
    """
    request = str(requested or "auto").strip().lower()
    if request == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return "cuda"
    if request == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
        return "mps"
    if request == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported training.device value: {requested!r}")

