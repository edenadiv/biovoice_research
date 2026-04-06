# Evaluation Protocol

## Decision Targets
- `target_bona_fide`
- `spoof`
- `wrong_speaker`
- optional `manual_review`

## Core Evaluation Views
- Speaker verification performance on target vs non-target trials
- Spoof detection performance on bona fide vs spoof probes
- Final three-way decision macro F1, balanced accuracy, accuracy, and confusion structure
- Majority-class baseline comparison for the final three-way task
- Decision-path summary showing how often the spoof gate dominates the final rule system
- Reliability diagnostics for saved probabilities
- Segment-level suspiciousness inspection for case studies
- Threshold sweep stability and joint-score geometry

## Alpha Review Focus
Alpha review is about methodological coherence and artifact quality, not state-of-the-art performance claims.

## Why Accuracy Alone Is Not Enough
The current real-data ASVspoof baseline is heavily imbalanced toward `spoof`. In that setting, a trivial always-`spoof` predictor can achieve deceptively high accuracy. For the joint three-way task, supervisors should therefore treat:

1. macro F1
2. balanced accuracy
3. classwise precision/recall/F1
4. confusion matrices
5. majority-baseline comparison

as the primary evidence, with raw accuracy only as supporting context.

## Threshold Selection Policy
- SV and spoof thresholds are selected on validation data only.
- The default optimization objective is macro F1 because the joint task is imbalanced.
- The spoof branch should also be trained with a validation monitor that respects class skew; the current baseline keeps the best spoof checkpoint by validation balanced accuracy (`val_metric`) rather than by validation loss alone.
- The saved threshold artifacts include:
  - the full threshold sweep table
  - the threshold heatmap
  - the selected threshold pair
  - a comparison between default and tuned thresholds on the evaluation split
- Test data should not be used to choose thresholds.

## Recommended Review Order
1. Check split manifests and leakage notes.
2. Review the majority-baseline comparison and classwise metrics table.
3. Review baseline training curves.
4. Review score distributions and confusion matrices.
5. Review the decision-path summary to see whether the spoof gate is overwhelming the final decision space.
6. Review the threshold sweep heatmap, threshold comparison table, and reliability diagram.
7. Review segment-level case studies only after the global figures make sense.
