"""Dataset loading utilities for utterance-level and trial-level experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from biovoice.data.manifests import load_manifest
from biovoice.data.preprocessing import preprocess_audio
from biovoice.utils.audio_io import load_audio


@dataclass(slots=True)
class TrialAudioBundle:
    """Preprocessed audio bundle for one enrollment-conditioned trial."""

    trial_id: str
    speaker_id: str
    probe_waveform: torch.Tensor
    enrollment_waveforms: list[torch.Tensor]
    probe_path: str
    enrollment_paths: list[str]
    label: str
    split: str
    speaker_match_label: int
    spoof_label: int
    metadata: dict[str, Any]


class UtteranceDataset(Dataset):
    """Dataset for SV and spoof branch training.

    The same dataset supports both branches so preprocessing stays identical.
    """

    def __init__(
        self,
        manifest: str | pd.DataFrame,
        preprocessing_config: dict[str, Any],
        split: str,
        speaker_to_index: dict[str, int] | None = None,
        only_bona_fide: bool = False,
    ) -> None:
        self.frame = load_manifest(manifest) if isinstance(manifest, (str, bytes, Path)) else manifest.copy()
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        if only_bona_fide:
            self.frame = self.frame[self.frame["spoof_label"] == 0].reset_index(drop=True)
        self.preprocessing_config = preprocessing_config
        self.speaker_to_index = speaker_to_index or {
            speaker: index for index, speaker in enumerate(sorted(self.frame["speaker_id"].unique()))
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        waveform, sample_rate = load_audio(row["path"])
        processed = preprocess_audio(waveform, sample_rate, self.preprocessing_config)
        return {
            "waveform": processed.waveform,
            "speaker_id": row["speaker_id"],
            "speaker_index": self.speaker_to_index[row["speaker_id"]],
            "spoof_label": int(row["spoof_label"]),
            "path": row["path"],
            "utterance_id": row["utterance_id"],
        }


class TrialDataset(Dataset):
    """Dataset for evaluation on enrollment-conditioned trials."""

    def __init__(self, manifest: str | pd.DataFrame, preprocessing_config: dict[str, Any], split: str) -> None:
        self.frame = load_manifest(manifest) if isinstance(manifest, (str, bytes, Path)) else manifest.copy()
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        self.preprocessing_config = preprocessing_config
        # Real evaluation manifests can reuse the same enrollment files across a
        # large number of adjacent trials. A small LRU cache avoids repeatedly
        # decoding and preprocessing identical audio without trying to keep the
        # entire corpus resident in memory.
        self.cache_size = int(preprocessing_config.get("inference_waveform_cache_size", 0))
        self._waveform_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frame)

    def _get_processed_waveform(self, path: str) -> torch.Tensor:
        """Load, preprocess, and optionally cache one audio path."""
        if self.cache_size > 0 and path in self._waveform_cache:
            waveform = self._waveform_cache.pop(path)
            self._waveform_cache[path] = waveform
            return waveform
        waveform, sample_rate = load_audio(path)
        processed = preprocess_audio(waveform, sample_rate, self.preprocessing_config).waveform
        if self.cache_size > 0:
            self._waveform_cache[path] = processed
            while len(self._waveform_cache) > self.cache_size:
                self._waveform_cache.popitem(last=False)
        return processed

    def __getitem__(self, index: int) -> TrialAudioBundle:
        row = self.frame.iloc[index]
        processed_probe = self._get_processed_waveform(row["probe_path"])
        enrollment_waveforms = []
        for path in row["enrollment_paths"]:
            enrollment_waveforms.append(self._get_processed_waveform(path))
        return TrialAudioBundle(
            trial_id=row["trial_id"],
            speaker_id=row["speaker_id"],
            probe_waveform=processed_probe,
            enrollment_waveforms=enrollment_waveforms,
            probe_path=row["probe_path"],
            enrollment_paths=list(row["enrollment_paths"]),
            label=row["label"],
            split=row["split"],
            speaker_match_label=int(row.get("speaker_match_label", row["label"] == "target_bona_fide")),
            spoof_label=int(row.get("spoof_label", row["label"] == "spoof")),
            metadata=row.to_dict(),
        )


def collate_utterances(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack utterance examples into a mini-batch."""
    waveforms = torch.stack([item["waveform"] for item in batch], dim=0)
    speaker_indices = torch.tensor([item["speaker_index"] for item in batch], dtype=torch.long)
    spoof_labels = torch.tensor([item["spoof_label"] for item in batch], dtype=torch.float32)
    return {
        "waveform": waveforms,
        "speaker_index": speaker_indices,
        "spoof_label": spoof_labels,
        "speaker_id": [item["speaker_id"] for item in batch],
        "path": [item["path"] for item in batch],
        "utterance_id": [item["utterance_id"] for item in batch],
    }
