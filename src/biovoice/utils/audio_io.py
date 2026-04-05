"""Audio input/output helpers with light fallbacks.

Torchaudio is the primary backend because the repository is PyTorch-first.
On some modern torchaudio builds, however, FLAC decoding is delegated to
``torchcodec``. Real ASVspoof runs therefore also try ``soundfile`` before
falling back to SciPy's WAV-only reader.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import wavfile

try:
    import torchaudio
except Exception:  # pragma: no cover - fallback is intentionally broad.
    torchaudio = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - optional dependency for FLAC support.
    sf = None


def load_audio(path: str | Path) -> Tuple[torch.Tensor, int]:
    """Load audio as a float tensor with shape ``[channels, samples]``.

    The fallback order is chosen to keep both demo and real-data execution
    reliable:
    1. ``torchaudio`` for the normal PyTorch-first path.
    2. ``soundfile`` for FLAC/other libsndfile-supported formats when
       torchaudio cannot decode without TorchCodec.
    3. SciPy for simple WAV fixtures used in tests and demo generation.
    """
    target = Path(path)
    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(str(target))
            return waveform.float(), sample_rate
        except Exception:
            # Newer torchaudio versions can delegate load() to torchcodec.
            # Falling back to SoundFile/SciPy avoids making real-data execution
            # depend on torchcodec being preinstalled.
            pass

    if sf is not None:
        try:
            waveform, sample_rate = sf.read(str(target), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(waveform.T.copy())
            return waveform, int(sample_rate)
        except Exception:
            # Some official ASVspoof FLAC files decode cleanly with FFmpeg even
            # when libsndfile reports an internal decoder error.
            pass

    if shutil.which("ffmpeg"):
        metadata = inspect_audio_metadata(target)
        process = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(target), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        waveform = np.frombuffer(process.stdout, dtype=np.float32)
        channels = max(int(metadata.get("num_channels", 1)), 1)
        if channels > 1:
            usable = waveform.size - (waveform.size % channels)
            waveform = waveform[:usable].reshape(-1, channels).T
        else:
            waveform = waveform[None, :]
        return torch.from_numpy(waveform.copy()), int(metadata["sample_rate"])

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


def inspect_audio_metadata(path: str | Path) -> dict[str, Any]:
    """Read lightweight audio metadata without decoding the full waveform.

    This helper is useful for large real-data corpora where duration and sample
    rate are needed for audit plots, but a full waveform pass over every file
    would dominate the run time. It intentionally returns a tiny plain dict so
    callers can serialize or merge the results easily.
    """
    target = Path(path)
    if sf is not None:
        try:
            info = sf.info(str(target))
            return {
                "sample_rate": int(info.samplerate),
                "num_frames": int(info.frames),
                "num_channels": int(info.channels),
            }
        except Exception:
            pass
    if shutil.which("ffprobe"):
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "json",
                str(target),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        payload = json.loads(process.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        duration_seconds = float((payload.get("format") or {}).get("duration", 0.0) or 0.0)
        sample_rate = int(stream.get("sample_rate") or 0)
        return {
            "sample_rate": sample_rate,
            "num_frames": int(round(duration_seconds * max(sample_rate, 1))),
            "num_channels": int(stream.get("channels") or 1),
        }
    sample_rate, waveform = wavfile.read(target)
    num_frames = int(waveform.shape[0]) if waveform.ndim > 1 else int(waveform.shape[-1])
    num_channels = int(waveform.shape[1]) if waveform.ndim > 1 else 1
    return {
        "sample_rate": int(sample_rate),
        "num_frames": num_frames,
        "num_channels": num_channels,
    }


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
