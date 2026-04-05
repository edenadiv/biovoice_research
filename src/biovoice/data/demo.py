"""Synthetic/demo corpus generation for smoke tests and alpha review.

The synthetic data is intentionally simple and clearly documented. It exists to
prove the repository runs end-to-end without making any claim that synthetic
results predict real-world performance.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biovoice.data.manifests import save_manifest, save_split_manifests
from biovoice.utils.audio_io import save_audio
from biovoice.utils.path_utils import resolve_path


def _speaker_frequency(speaker_index: int) -> float:
    return 110.0 + 18.0 * speaker_index


def _make_bona_fide_waveform(speaker_index: int, utterance_index: int, duration_seconds: float, sample_rate: int) -> torch.Tensor:
    length = int(duration_seconds * sample_rate)
    time = torch.linspace(0.0, duration_seconds, steps=length)
    base = _speaker_frequency(speaker_index)
    phase = 0.05 * utterance_index
    envelope = 0.5 + 0.5 * torch.sin(2 * math.pi * 1.5 * time + phase)
    harmonic_1 = torch.sin(2 * math.pi * base * time + phase)
    harmonic_2 = 0.5 * torch.sin(2 * math.pi * 2 * base * time + phase)
    harmonic_3 = 0.25 * torch.sin(2 * math.pi * 3 * base * time + phase)
    formant_like = 0.15 * torch.sin(2 * math.pi * (base / 4.0) * time)
    waveform = envelope * (harmonic_1 + harmonic_2 + harmonic_3) + formant_like
    waveform += 0.01 * torch.randn_like(waveform)
    return waveform.unsqueeze(0).clamp(-1.0, 1.0)


def _make_spoof_waveform(bona_fide: torch.Tensor) -> torch.Tensor:
    signal = bona_fide.squeeze(0)
    downsampled = torch.nn.functional.interpolate(
        signal.unsqueeze(0).unsqueeze(0),
        scale_factor=0.92,
        mode="linear",
        align_corners=False,
    ).squeeze()
    restored = torch.nn.functional.interpolate(
        downsampled.unsqueeze(0).unsqueeze(0),
        size=signal.numel(),
        mode="linear",
        align_corners=False,
    ).squeeze()
    synthetic_texture = 0.015 * torch.sign(torch.sin(torch.linspace(0.0, 80.0, steps=signal.numel())))
    spoof = 0.85 * restored + synthetic_texture
    return spoof.unsqueeze(0).clamp(-1.0, 1.0)


def _split_speakers(speaker_ids: list[str]) -> dict[str, list[str]]:
    if len(speaker_ids) < 4:
        raise ValueError("At least four speakers are required so the test split can include wrong-speaker trials.")
    test_count = 2
    remaining = len(speaker_ids) - test_count
    val_count = 1 if remaining > 2 else 0
    train_count = len(speaker_ids) - test_count - val_count
    if train_count < 2 and val_count > 0:
        val_count = 0
        train_count = len(speaker_ids) - test_count
    train = speaker_ids[:train_count]
    val = speaker_ids[train_count : train_count + val_count]
    test = speaker_ids[train_count + val_count :]
    return {"train": train, "val": val, "test": test}


def generate_demo_dataset(config: dict[str, Any]) -> dict[str, Path]:
    """Generate synthetic audio and leakage-safe manifests."""
    data_cfg = config["data"]
    sample_rate = int(data_cfg["sample_rate"])
    root = resolve_path(data_cfg["demo_root"])
    audio_root = root / "audio"
    manifests_root = root / "manifests"
    splits_root = manifests_root / "splits"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    speaker_ids = [f"spk_{index:03d}" for index in range(int(data_cfg["synthetic_speakers"]))]
    split_map = _split_speakers(speaker_ids)

    utterance_rows: list[dict[str, Any]] = []
    bona_by_speaker: dict[str, list[dict[str, Any]]] = {}
    spoof_by_speaker: dict[str, list[dict[str, Any]]] = {}

    for speaker_index, speaker_id in enumerate(speaker_ids):
        speaker_dir = audio_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        split = next(name for name, speakers in split_map.items() if speaker_id in speakers)
        bona_entries: list[dict[str, Any]] = []
        spoof_entries: list[dict[str, Any]] = []
        for utterance_index in range(int(data_cfg["synthetic_utterances_per_speaker"])):
            duration_seconds = 2.5 + 0.15 * utterance_index
            source_recording_id = f"{speaker_id}_src_{utterance_index:03d}"
            bona_waveform = _make_bona_fide_waveform(speaker_index, utterance_index, duration_seconds, sample_rate)
            spoof_waveform = _make_spoof_waveform(bona_waveform)

            bona_path = speaker_dir / f"{source_recording_id}_bona.wav"
            spoof_path = speaker_dir / f"{source_recording_id}_spoof.wav"
            save_audio(bona_path, bona_waveform, sample_rate)
            save_audio(spoof_path, spoof_waveform, sample_rate)

            bona_entry = {
                "utterance_id": f"{speaker_id}_utt_{utterance_index:03d}_bona",
                "speaker_id": speaker_id,
                "path": str(bona_path),
                "split": split,
                "spoof_label": 0,
                "duration_seconds": duration_seconds,
                "source_recording_id": source_recording_id,
            }
            spoof_entry = {
                "utterance_id": f"{speaker_id}_utt_{utterance_index:03d}_spoof",
                "speaker_id": speaker_id,
                "path": str(spoof_path),
                "split": split,
                "spoof_label": 1,
                "duration_seconds": duration_seconds,
                "source_recording_id": source_recording_id,
            }
            bona_entries.append(bona_entry)
            spoof_entries.append(spoof_entry)
            utterance_rows.extend([bona_entry, spoof_entry])
        bona_by_speaker[speaker_id] = bona_entries
        spoof_by_speaker[speaker_id] = spoof_entries

    utterance_frame = pd.DataFrame(utterance_rows)
    save_manifest(utterance_frame, manifests_root / "utterances.csv")
    save_split_manifests(utterance_frame, splits_root, "utterances")

    trial_rows: list[dict[str, Any]] = []
    enrollment_count = int(data_cfg["enrollment_count"])
    trial_counter = 0
    for split, speakers in split_map.items():
        for speaker_id in speakers:
            bona_entries = bona_by_speaker[speaker_id]
            spoof_entries = spoof_by_speaker[speaker_id]
            enrollment = bona_entries[:enrollment_count]
            enrollment_paths = [item["path"] for item in enrollment]
            enrollment_sources = [item["source_recording_id"] for item in enrollment]
            probe_candidates = bona_entries[enrollment_count:]
            for probe in probe_candidates[: int(data_cfg["synthetic_probe_trials_per_speaker"])]:
                trial_rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": probe["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "target_bona_fide",
                        "split": split,
                        "speaker_match_label": 1,
                        "spoof_label": 0,
                        "probe_source_recording_id": probe["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

            for spoof in spoof_entries[enrollment_count : enrollment_count + int(data_cfg["synthetic_probe_trials_per_speaker"])]:
                if spoof["source_recording_id"] in enrollment_sources:
                    continue
                trial_rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": spoof["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "spoof",
                        "split": split,
                        "speaker_match_label": 1,
                        "spoof_label": 1,
                        "probe_source_recording_id": spoof["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

            other_speakers = [candidate for candidate in speakers if candidate != speaker_id]
            if other_speakers:
                imposter_speaker = other_speakers[0]
                imposter_probe = bona_by_speaker[imposter_speaker][enrollment_count]
                trial_rows.append(
                    {
                        "trial_id": f"trial_{trial_counter:05d}",
                        "speaker_id": speaker_id,
                        "claim_id": speaker_id,
                        "probe_path": imposter_probe["path"],
                        "enrollment_paths": enrollment_paths,
                        "label": "wrong_speaker",
                        "split": split,
                        "speaker_match_label": 0,
                        "spoof_label": 0,
                        "probe_source_recording_id": imposter_probe["source_recording_id"],
                        "enrollment_source_recording_ids": "|".join(enrollment_sources),
                    }
                )
                trial_counter += 1

    trial_frame = pd.DataFrame(trial_rows)
    save_manifest(trial_frame, manifests_root / "trials.csv")
    save_split_manifests(trial_frame, splits_root, "trials")

    assumptions = {
        "description": "Synthetic corpus for smoke tests only. Voices are harmonic toy signals rather than natural speech.",
        "leakage_policy": "Enrollment and probe files never share the same source_recording_id within a trial.",
        "label_semantics": {
            "target_bona_fide": "Probe matches the claimed speaker and is not spoofed.",
            "spoof": "Probe is a synthetic manipulation of the claimed speaker.",
            "wrong_speaker": "Probe is bona fide but from a different speaker.",
        },
    }
    (manifests_root / "synthetic_assumptions.json").write_text(pd.Series(assumptions).to_json(indent=2), encoding="utf-8")
    return {
        "audio_root": audio_root,
        "utterance_manifest": manifests_root / "utterances.csv",
        "trial_manifest": manifests_root / "trials.csv",
        "split_manifest_dir": splits_root,
    }
