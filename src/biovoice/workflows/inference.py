"""Inference-time helpers for enrollment-conditioned scoring and explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from biovoice.data.enrollment import aggregate_embeddings
from biovoice.data.loading import TrialDataset
from biovoice.explain.case_analysis import build_case_analysis
from biovoice.explain.feature_attribution import top_feature_contributors
from biovoice.explain.reason_generator import generate_reasons
from biovoice.explain.segment_reasoning import rank_suspicious_segments
from biovoice.features.acoustic_features import extract_acoustic_features
from biovoice.features.biometric_features import compare_feature_dicts
from biovoice.features.temporal_features import extract_temporal_features
from biovoice.models.model_factory import build_anti_spoof_model, build_speaker_model
from biovoice.models.segment_model import score_segments
from biovoice.training.checkpointing import load_checkpoint
from biovoice.training.device import resolve_device
from biovoice.utils.serialization import save_frame, save_json
from biovoice.viz.explainability_plots import (
    plot_feature_contributions,
    plot_segment_score_timeline,
    plot_waveform_with_segments,
)


def _mean_feature_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of feature dictionaries into one speaker summary profile."""
    keys = sorted({key for item in items for key in item.keys()})
    result = {}
    for key in keys:
        values = [item[key] for item in items if key in item]
        result[key] = float(np.mean(values))
    return result


def build_trial_predictions(
    config: dict[str, Any],
    run_paths: Any,
    sv_checkpoint: Path,
    spoof_checkpoint: Path,
    split: str = "test",
    logger: Any | None = None,
) -> pd.DataFrame:
    """Run enrollment-conditioned inference and save per-trial explanation files."""
    device = resolve_device(config["training"].get("device", "auto"))
    from biovoice.data.manifests import load_manifest

    utterance_manifest = load_manifest(config["data"]["utterance_manifest_path"])
    train_speaker_count = int(
        utterance_manifest.loc[
            utterance_manifest["split"] == config["data"]["train_split"], "speaker_id"
        ].nunique()
    )
    speaker_model = build_speaker_model(config, num_speakers=max(train_speaker_count, 2))
    spoof_model = build_anti_spoof_model(config)
    load_checkpoint(sv_checkpoint, speaker_model)
    load_checkpoint(spoof_checkpoint, spoof_model)
    speaker_model.to(device)
    spoof_model.to(device)
    speaker_model.eval()
    spoof_model.eval()

    dataset = TrialDataset(config["data"]["trial_manifest_path"], config["preprocessing"], split=split)
    rows: list[dict[str, Any]] = []
    sample_rate = int(config["data"]["sample_rate"])
    enrollment_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    probe_cache: dict[str, dict[str, Any]] = {}
    max_saved_cases = int(config["explainability"].get("max_saved_case_files", 100))
    progress_every = int(config["evaluation"].get("prediction_progress_every", 500))
    for trial_index, bundle in enumerate(dataset, start=1):
        enrollment_key = tuple(str(path) for path in bundle.enrollment_paths)
        save_case_details = trial_index <= max_saved_cases
        if enrollment_key in enrollment_cache:
            enrollment_template = enrollment_cache[enrollment_key]["template"]
        else:
            with torch.no_grad():
                enrollment_embeddings = []
                for waveform in bundle.enrollment_waveforms:
                    embedding = speaker_model.encoder(waveform.unsqueeze(0).to(device))
                    enrollment_embeddings.append(embedding.squeeze(0))
                enrollment_template = aggregate_embeddings(torch.stack(enrollment_embeddings, dim=0))
            enrollment_cache[enrollment_key] = {"template": enrollment_template, "feature_mean": None}

        enrollment_feature_mean: dict[str, float] | None = enrollment_cache[enrollment_key]["feature_mean"]
        if save_case_details and enrollment_feature_mean is None:
            enrollment_features = [
                {**extract_acoustic_features(waveform, sample_rate), **extract_temporal_features(waveform, sample_rate)}
                for waveform in bundle.enrollment_waveforms
            ]
            enrollment_feature_mean = _mean_feature_dicts(enrollment_features)
            enrollment_cache[enrollment_key]["feature_mean"] = enrollment_feature_mean

        if bundle.probe_path in probe_cache:
            cached_probe = probe_cache[bundle.probe_path]
            probe_embedding = cached_probe["embedding"]
            spoof_probability = cached_probe["spoof_probability"]
        else:
            with torch.no_grad():
                probe_embedding = speaker_model.encoder(bundle.probe_waveform.unsqueeze(0).to(device)).squeeze(0)
                spoof_output = spoof_model(bundle.probe_waveform.unsqueeze(0).to(device))
                spoof_probability = float(spoof_output["probability"].item())
            probe_cache[bundle.probe_path] = {
                "embedding": probe_embedding,
                "spoof_probability": spoof_probability,
                "features": None,
            }
        probe_features: dict[str, float] | None = probe_cache[bundle.probe_path]["features"]
        if save_case_details and probe_features is None:
            probe_features = {
                **extract_acoustic_features(bundle.probe_waveform, sample_rate),
                **extract_temporal_features(bundle.probe_waveform, sample_rate),
            }
            probe_cache[bundle.probe_path]["features"] = probe_features

        with torch.no_grad():
            sv_score = float(torch.nn.functional.cosine_similarity(probe_embedding.unsqueeze(0), enrollment_template.unsqueeze(0)).item())
        feature_deltas: dict[str, float] = {}
        if save_case_details and enrollment_feature_mean is not None and probe_features is not None:
            feature_deltas = compare_feature_dicts(enrollment_feature_mean, probe_features)
        segment_frame = pd.DataFrame()
        suspicious_segments = pd.DataFrame()
        if save_case_details:
            segment_rows, _ = score_segments(
                bundle.probe_waveform,
                sample_rate,
                config["segmentation"],
                speaker_model.encoder,
                enrollment_template,
                spoof_model,
            )
            segment_frame = pd.DataFrame(segment_rows)
            suspicious_segments = rank_suspicious_segments(
                segment_frame,
                top_k=int(config["explainability"]["suspicious_segment_count"]),
            )
        from biovoice.evaluation.thresholding import final_decision

        decision = final_decision(
            sv_score=sv_score,
            spoof_probability=spoof_probability,
            sv_threshold=float(config["evaluation"]["sv_threshold"]),
            spoof_threshold=float(config["evaluation"]["spoof_threshold"]),
            manual_review_margin=float(config["evaluation"]["manual_review_margin"]),
        )
        top_features = (
            top_feature_contributors(feature_deltas, top_k=int(config["explainability"]["explanation_feature_count"]))
            if save_case_details
            else []
        )
        reasons = generate_reasons(decision, sv_score, spoof_probability, top_features, suspicious_segments)
        if save_case_details:
            case_payload = build_case_analysis(bundle.trial_id, decision, reasons, suspicious_segments, top_features)
            save_json(case_payload, run_paths.explainability / f"{bundle.trial_id}_case.json")
            save_frame(segment_frame, run_paths.explainability / f"{bundle.trial_id}_segments.csv")

        if len(rows) == 0:
            plot_waveform_with_segments(
                bundle.probe_waveform,
                sample_rate,
                suspicious_segments,
                run_paths.plots / "waveform_with_suspicious_segments.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_segment_score_timeline(
                segment_frame,
                "spoof_probability",
                "Spoof Score Over Time",
                run_paths.plots / "spoof_score_over_time.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_segment_score_timeline(
                segment_frame,
                "speaker_similarity",
                "Speaker Similarity Over Time",
                run_paths.plots / "speaker_similarity_over_time.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )
            plot_feature_contributions(
                top_features,
                run_paths.plots / "feature_contributions.png",
                dpi=config["plotting"]["dpi"],
                style=config["plotting"]["style"],
            )

        rows.append(
            {
                "trial_id": bundle.trial_id,
                "speaker_id": bundle.speaker_id,
                "label": bundle.label,
                "speaker_match_label": bundle.speaker_match_label,
                "spoof_label": bundle.spoof_label,
                "sv_score": sv_score,
                "spoof_probability": spoof_probability,
                "final_decision": decision,
                "probe_path": bundle.probe_path,
                "probe_duration_seconds": bundle.probe_waveform.shape[-1] / sample_rate,
                **feature_deltas,
                "reason_1": reasons[0] if reasons else "",
            }
        )
        if logger is not None and trial_index % max(progress_every, 1) == 0:
            logger.info("Scored %d/%d trials for split '%s'.", trial_index, len(dataset), split)
    return pd.DataFrame(rows)
