# Plots Guide

## Mandatory Alpha Figures
- Class balance plot
  What it shows: distribution of final trial labels.
  Why it matters: helps frame all later metrics.
  How to read it: large imbalance means accuracy should not be read in isolation.
- Duration histogram
  What it shows: distribution of clip durations after preprocessing.
  Why it matters: very short probes are harder for both SV and spoof branches.
  How to read it: a broad range suggests duration-dependent robustness checks are important.
- Train/validation loss curves
  What it shows: optimization traces for the baseline branches.
  Why it matters: confirms that the training runs completed and whether they overfit quickly.
  How to read it: stable, gradually improving curves are healthier than noisy, diverging curves.
- ROC curves
  What it shows: threshold-free ranking quality for SV and spoof branches.
  Why it matters: useful before fixing one operating threshold.
  How to read it: curves closer to the upper-left corner are stronger.
- PR curves
  What it shows: precision-recall trade-off for binary subtasks.
  Why it matters: complements ROC when class counts differ.
  How to read it: high precision sustained at useful recall values is better.
- DET curves
  What it shows: false-positive versus false-negative trade-offs.
  Why it matters: often easier to interpret in biometric settings.
  How to read it: curves closer to the origin are better.
- Confusion matrix
  What it shows: raw counts of final decision outcomes.
  Why it matters: directly exposes which error mode dominates.
  How to read it: look for large off-diagonal cells before reading the scalar accuracy.
- Normalized confusion matrix
  What it shows: row-normalized classwise recalls.
  Why it matters: easier to compare behavior across uneven classes.
  How to read it: diagonal values near one are desirable.
- Target vs non-target score distributions
  What it shows: SV score separation.
  Why it matters: reveals whether the SV branch alone is credible.
  How to read it: heavy overlap means fusion is likely necessary.
- Spoof vs bona fide score distributions
  What it shows: spoof probability separation.
  Why it matters: reveals whether the spoof branch alone is credible.
  How to read it: overlap means spoof-only screening is insufficient.
- Reliability diagram
  What it shows: predicted spoof probability versus observed frequency.
  Why it matters: probability outputs should not be trusted blindly.
  How to read it: the closer to the diagonal, the better calibrated the branch.
- Threshold sweep heatmap
  What it shows: decision accuracy across a grid of SV and spoof thresholds.
  Why it matters: exposes brittle versus stable operating regions.
  How to read it: broad bright regions are better than one isolated optimum.
- Score scatter plot
  What it shows: SV score versus spoof probability by class.
  Why it matters: visually explains why late fusion may help.
  How to read it: classes occupying distinct regions support the fusion design.
- Ablation summary bar chart
  What it shows: comparison across the four main alpha experiment modes.
  Why it matters: summarizes whether fusion adds value over individual branches.
  How to read it: treat differences on demo data as behavior checks, not publication claims.
- Waveform with suspicious segments
  What it shows: highlighted suspicious spans on the probe waveform.
  Why it matters: gives supervisors a tangible time-local view of the explanation.
  How to read it: compare with the two temporal score plots rather than reading it alone.
- Spoof score over time
  What it shows: segment-level spoof probability trajectory.
  Why it matters: reveals localized suspicious peaks.
  How to read it: sharp peaks can suggest partial or localized manipulation.
- Speaker similarity over time
  What it shows: segment-level biometric consistency trajectory.
  Why it matters: reveals whether the claimed-speaker match is stable throughout the probe.
  How to read it: localized drops can point to wrong-speaker or manipulated regions.

## How Supervisors Should Read These Plots
- Look first for obvious pathologies such as collapsed score distributions or highly unstable loss curves.
- Use score distributions and confusion matrices together rather than in isolation.
- Use the threshold sweep heatmap before arguing that one operating threshold is well justified.
- Treat explainability plots as inspection aids, not as standalone proof.

## Common Plot Pitfalls
- A visually clean demo-data plot does not imply real-world generalization.
- Small test sets can make curves and histograms look deceptively sharp.
- Calibration plots can look unstable when only a few points populate each bin.
- Explainability timelines should not be treated as causal truth about the waveform.
