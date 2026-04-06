# Supervisor Review Report

## Project Summary
Leakage-safe real-data alpha baseline combining speaker verification, spoof detection, validation-tuned fusion logic, and interpretable mismatch analysis.

## How To Read This Report
Start with the alpha-exit evidence, then inspect the key metrics, then open the mandatory figures and the walkthrough notebook. Treat the current run numbers as measured baseline evidence rather than broad real-world performance claims.

## Alpha Exit Criteria Evidence
- Leakage-safe ASVspoof 2019/2021 LA data staged and consumed end-to-end.
- Baseline SV run completed and saved history/checkpoint artifacts.
- Baseline spoof run completed and saved history/checkpoint artifacts.
- Fusion evaluation completed with saved predictions, metrics, and mandatory figures.
- Supervisor notebook can read saved artifacts from outputs.

## Key Metrics
- Joint Macro F1: 0.442
- Joint Balanced Accuracy: 0.462
- Joint Accuracy: 0.691
- Majority-Baseline Accuracy: 0.818
- SV EER: 0.312
- Spoof ROC-AUC: 0.649
- Calibration ECE: 0.103

## Dataset Review
- Dataset mode: asvspoof2021_la
- Dataset name: ASVspoof2021_LA
- Split strategy: official_protocol
- Speaker-disjoint requirement: True
- Speaker-disjoint status: pass
- Speaker-disjoint violations: 0
- Leakage status: pass
- Trial leakage violations: 0
- Evaluation trial counts by label: {'spoof': 133360, 'target_bona_fide': 14816, 'wrong_speaker': 14816}
- Validation trial counts by label: {'spoof': 22296, 'target_bona_fide': 2548, 'wrong_speaker': 2548}
- Quality scan mode: header_plus_sample
- Quality measurement counts: {'header_only': 117461, 'waveform': 4000}
- Enrollment policy: Enrollment candidates are ranked with a seed-stable order over distinct source recordings rather than relying on archive file order.
- Wrong-speaker policy: Wrong-speaker claims reuse bona fide probes under seed-stable alternative claims while excluding any enrollment source that overlaps the probe.

## Metric Interpretation
- Treat SV metrics as biometric-consistency evidence only. Strong spoofers can still fool this branch.
- Treat spoof metrics as artifact-detection evidence only. Weak spoof separation will usually bottleneck the final three-way decision.
- For the imbalanced three-way task, macro F1 and balanced accuracy are more honest than raw accuracy alone.
- Calibration and threshold-selection artifacts matter because the final decision is thresholded, not purely ranking-based.
- Current joint accuracy is below the trivial always-spoof baseline on this class distribution.

## Baseline Comparison
- Majority baseline (`always spoof`) accuracy is 0.818 with macro F1 0.300 and balanced accuracy 0.333.
- SV-only reaches accuracy 0.129, macro F1 0.172, and balanced accuracy 0.472; spoof-only reaches accuracy 0.766, macro F1 0.363, and balanced accuracy 0.385.
- Default fusion reaches accuracy 0.775, macro F1 0.429, and balanced accuracy 0.416; tuned fusion reaches accuracy 0.691, macro F1 0.442, and balanced accuracy 0.462.
- Tuned fusion changes relative to default thresholds: accuracy -0.083, macro F1 +0.013, balanced accuracy +0.046.
- Because the class distribution is spoof-heavy, the majority baseline is a mandatory comparison and plain accuracy should not be interpreted alone.

## Threshold Selection
- Validation split used for tuning: val.
- Optimization objective: macro_f1 with selected SV threshold 0.620 and spoof threshold 0.690.
- Default thresholds were SV 0.550 and spoof 0.500; `used_tuned_thresholds` was True.

## Classwise Results
- spoof: precision 0.862, recall 0.778, F1 0.818, support 133360.
- target_bona_fide: precision 0.171, recall 0.311, F1 0.221, support 14816.
- wrong_speaker: precision 0.278, recall 0.297, F1 0.287, support 14816.

## Decision Path Summary
- The spoof gate fired on 0.738 of trials (120258 / 162992), with a true-spoof fraction of 0.862 inside that path.
- After the spoof gate stayed open, the SV accept path covered 0.165 of trials and still contained a true-spoof fraction of 0.750, showing how many spoofs slip past branch B.
- The SV reject path after spoof rejection covered 0.097 of trials with a true-spoof fraction of 0.600.

## Error Summary
- spoof -> target_bona_fide: 20172 trials (0.151 of that true class).
- spoof -> wrong_speaker: 9498 trials (0.071 of that true class).
- wrong_speaker -> spoof: 8284 trials (0.559 of that true class).

## How To Interpret This Real-Data Baseline
- These metrics and figures are measured evidence for the current dataset and protocol only.
- This is still alpha-level evidence because the baselines are compact and robustness analysis remains limited.
- Do not overclaim real-world generalization, deployment readiness, or causal explainability from this run alone.

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
- plots/spoof_probability_by_true_label.png
- plots/spoof_roc.png
- plots/spoof_score_over_time.png
- plots/spoof_vs_bonafide_scores.png
- plots/sv_det.png
- plots/sv_loss_curves.png
- plots/sv_pr.png
- plots/sv_roc.png
- plots/sv_score_by_true_label.png
- plots/target_vs_non_target_scores.png
- plots/threshold_sweep_heatmap.png
- plots/waveform_with_suspicious_segments.png

## Artifact Map
- Artifact index: reports/artifact_index.md
- Plot inventory: reports/plot_inventory.md
- Metric summary: reports/metric_summary.md
- Prediction table: tables/predictions.csv
- Baseline comparison table: reports/baseline_comparison.md
- Classwise metrics table: reports/joint_classwise_metrics.md
- Decision-path summary table: reports/decision_path_summary.md
- Threshold comparison table: reports/threshold_comparison.md
- Notebook-ready walkthrough artifacts live under: outputs/runs/20260405_200907_fusion_plus_interpretable_features

## Limitations
- Baselines are research baselines, not production systems.
- No real-world robustness claims should be made until broader evaluation is completed.
- Explainability outputs are supporting evidence, not proof of causality.
- Short or noisy audio may weaken both speaker-verification and spoof judgments.
- Stronger pretrained backbones and larger-scale benchmarking are future work.

## Recommended Next Steps
- Improve the spoof branch first, because it remains the main bottleneck in the final three-way decision.
- Use the threshold comparison and classwise metrics when discussing the run, not accuracy alone.
- Inspect the threshold heatmap and the per-class score plots before proposing any stronger decision rule.
- Treat the current numbers as measured ASVspoof baseline evidence and avoid broader real-world claims until robustness studies are complete.