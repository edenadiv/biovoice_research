# Supervisor Review Report

## Project Summary
Enrollment-conditioned alpha baseline combining speaker verification, spoof detection, fusion logic, and interpretable mismatch analysis.

## How To Read This Report
Start with the alpha-exit evidence, then inspect the key metrics, then review the figure inventory and the walkthrough notebook. Treat these numbers as real-data ASVspoof alpha-baseline evidence for this evaluated protocol only, not as broader real-world performance claims.

## Alpha Exit Criteria Evidence
- ASVspoof 2019/2021 LA data staged and consumed end-to-end.
- Baseline SV run completed and saved history/checkpoint artifacts.
- Baseline spoof run completed and saved history/checkpoint artifacts.
- Fusion evaluation completed with saved predictions, metrics, and mandatory figures.
- Supervisor notebook can read saved artifacts from outputs.

## Key Metrics
- SV EER: 0.327
- Spoof ROC-AUC: 0.649
- Joint Accuracy: 0.775
- Calibration ECE: 0.103

## Dataset Review
- Dataset mode: asvspoof2021_la
- Dataset name: ASVspoof2021_LA
- Split strategy: official_protocol
- Speaker-disjoint requirement: True
- Speaker-disjoint violations: 0
- Trial leakage violations: 0
- Trial counts by label: {'spoof': 133360, 'target_bona_fide': 14816, 'wrong_speaker': 14816}

## Metric Interpretation
- Treat SV metrics as biometric-consistency evidence only. Strong spoofers can still fool this branch.
- Treat spoof metrics as artifact-detection evidence only. Bona fide wrong speakers can still appear non-spoofed.
- Joint metrics matter most for the research question because they reflect the final three-way decision.
- Calibration metrics matter when thresholds and probabilities are used in reports or manual review rules.

## Mandatory Figures
- plots/ablation_summary.png
- plots/class_balance.png
- plots/confusion_matrix.png
- plots/duration_histogram.png
- plots/feature_contributions.png
- plots/normalized_confusion_matrix.png
- plots/reliability_diagram.png
- plots/score_scatter.png
- plots/speaker_similarity_over_time.png
- plots/spoof_det.png
- plots/spoof_loss_curves.png
- plots/spoof_pr.png
- plots/spoof_roc.png
- plots/spoof_score_over_time.png
- plots/spoof_vs_bonafide_scores.png
- plots/sv_det.png
- plots/sv_loss_curves.png
- plots/sv_pr.png
- plots/sv_roc.png
- plots/target_vs_non_target_scores.png
- plots/threshold_sweep_heatmap.png
- plots/waveform_with_suspicious_segments.png

## Artifact Map
- Curated artifact folder: `artifacts/latest_run/`
- Metrics JSON: [metrics.json](metrics.json)
- Dataset review: [dataset_review.json](dataset_review.json)
- Metric summary table: [metric_summary.csv](metric_summary.csv)
- Plot inventory: [plot_inventory.md](plot_inventory.md)
- Prediction preview: [predictions_head.csv](predictions_head.csv)
- Source run directory: `outputs/runs/20260405_200907_fusion_plus_interpretable_features`
- Full plots, notebooks, checkpoints, and complete predictions remain in the local ignored `outputs/` tree.

## Limitations
- Baselines are research baselines, not production systems.
- No real-world robustness claims should be made until broader evaluation is completed.
- Explainability outputs are supporting evidence, not proof of causality.
- Short or noisy audio may weaken both speaker-verification and spoof judgments.
- Stronger pretrained backbones and larger-scale benchmarking are future work.

## Recommended Next Steps
- Inspect the threshold heatmap to choose a more defensible operating region before claiming a decision policy.
- Use the plot inventory when presenting figures to supervisors so every figure is paired with an interpretation note.
- Treat the current numbers as measured ASVspoof baseline evidence and avoid broader real-world claims until robustness studies are complete.
