# Evaluation Protocol

## Decision Targets
- `target_bona_fide`
- `spoof`
- `wrong_speaker`
- optional `manual_review`

## Core Evaluation Views
- Speaker verification performance on target vs non-target trials
- Spoof detection performance on bona fide vs spoof probes
- Final three-way decision accuracy and confusion structure
- Reliability diagnostics for saved probabilities
- Segment-level suspiciousness inspection for case studies
- Threshold sweep stability and joint-score geometry

## Alpha Review Focus
Alpha review is about methodological coherence and artifact quality, not state-of-the-art performance claims.

## Recommended Review Order
1. Check split manifests and leakage notes.
2. Review baseline training curves.
3. Review score distributions and confusion matrices.
4. Review the threshold sweep heatmap and reliability diagram.
5. Review segment-level case studies only after the global figures make sense.
