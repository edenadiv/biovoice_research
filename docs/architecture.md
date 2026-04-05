# Architecture

## Main Branches
- Speaker verification branch: learns speaker-discriminative embeddings from log-Mel inputs and compares probe embeddings with aggregated enrollment embeddings.
- Anti-spoof branch: scores utterances and segments for synthetic-manipulation likelihood.
- Fusion branch: combines SV score, spoof score, and interpretable mismatch evidence.
- Explainability branch: ranks suspicious segments, reports top feature deltas, and generates human-readable explanation cues.

## Data Flow
1. Load manifests and validate split assumptions.
2. Preprocess audio with sample-rate normalization, mono conversion, trimming, and length control.
3. Train SV and spoof baselines.
4. Run enrollment-conditioned trial inference.
5. Generate metrics, figures, reports, and notebook-ready artifacts.

## Artifact Philosophy
Every important step writes explicit artifacts so supervisors can inspect saved evidence without rerunning the entire experiment.
