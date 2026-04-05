"""Audio input/output helpers with light fallbacks.

Torchaudio is the primary backend because the repository is PyTorch-first.
SciPy is used as a pragmatic fallback for demo-data generation and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

try:
    import torchaudio
except Exception:  # pragma: no cover - fallback is intentionally broad.
    torchaudio = None


def load_audio(path: str | Path) -> Tuple[torch.Tensor, int]:
    """Load audio as a float tensor with shape ``[channels, samples]``."""
    target = Path(path)
    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(str(target))
            return waveform.float(), sample_rate
        except Exception:
            # Newer torchaudio versions can delegate load() to torchcodec.
            # Falling back to SciPy avoids making demo execution depend on it.
            pass

    sample_rate, waveform = wavfile.read(target)
    original_dtype = waveform.dtype
    waveform = waveform.astype(np.float32)
    if original_dtype.kind in {"i", "u"}:
        waveform = waveform / np.iinfo(original_dtype).max
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    else:
        waveform = waveform.T
    return torch.from_numpy(waveform), int(sample_rate)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio to disk, using torchaudio when available."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    waveform = waveform.detach().cpu().float()
    if torchaudio is not None:
        try:
            torchaudio.save(str(target), waveform, sample_rate)
            return
        except Exception:
            # Newer torchaudio versions can delegate save() to torchcodec.
            # Falling back to SciPy keeps demo-data generation self-contained.
            pass
    clipped = waveform.clamp(-1.0, 1.0).numpy().T
    wavfile.write(target, sample_rate, (clipped * 32767).astype(np.int16))


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average all channels into a single channel."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_audio(waveform: torch.Tensor, original_rate: int, target_rate: int) -> torch.Tensor:
    """Resample audio with torchaudio when available and linear fallback otherwise."""
    if original_rate == target_rate:
        return waveform
    if torchaudio is not None:
        return torchaudio.functional.resample(waveform, original_rate, target_rate)
    target_length = int(round(waveform.shape[-1] * target_rate / original_rate))
    return F.interpolate(waveform.unsqueeze(0), size=target_length, mode="linear", align_corners=False).squeeze(0)


def pad_or_truncate(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    """Pad with zeros or truncate so all training examples share a common length."""
    current = waveform.shape[-1]
    if current == target_num_samples:
        return waveform
    if current > target_num_samples:
        return waveform[..., :target_num_samples]
    pad = target_num_samples - current
    return F.pad(waveform, (0, pad))


def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """Normalize signal loudness using a simple RMS target."""
    rms = waveform.pow(2).mean().sqrt().clamp_min(1e-6)
    return waveform * (target_rms / rms)
