# Metrics Explained

This document explains what the repository reports, why each metric matters, and how a supervisor should interpret it in an alpha-review setting.

## Classification Metrics
- Accuracy
  What it is: fraction of examples that were predicted correctly.
  Why it matters: gives a quick overall sense of whether the decision pipeline is working.
  Higher or lower: higher is better.
  Common pitfall: accuracy can look acceptable even when one class is being ignored.
- Precision
  What it is: fraction of predicted positives that are truly positive.
  Why it matters: useful when false alarms are costly.
  Higher or lower: higher is better.
  Common pitfall: precision can improve simply by predicting positive less often.
- Recall
  What it is: fraction of actual positives that were recovered.
  Why it matters: useful when missed detections are costly.
  Higher or lower: higher is better.
  Common pitfall: recall can improve by over-predicting the positive class.
- F1
  What it is: harmonic mean of precision and recall.
  Why it matters: gives one summary value when both false positives and false negatives matter.
  Higher or lower: higher is better.
  Common pitfall: F1 still hides which error type dominates.
- Macro F1
  What it is: unweighted average of classwise F1 values.
  Why it matters: better than plain accuracy when classes are imbalanced.
  Higher or lower: higher is better.
  Common pitfall: small rare classes can make macro F1 noisy.
- Weighted F1
  What it is: class-frequency-weighted average of classwise F1 values.
  Why it matters: keeps rare classes from dominating the summary.
  Higher or lower: higher is better.
  Common pitfall: can still conceal poor performance on the smallest class.
- Balanced accuracy
  What it is: average recall across classes.
  Why it matters: a stronger default than accuracy for skewed datasets.
  Higher or lower: higher is better.
  Common pitfall: says nothing about precision.

## Speaker Verification Metrics
- Target vs non-target score distributions
  What it is: histograms of SV scores for genuine and impostor trials.
  Why it matters: the easiest way to see whether one threshold could plausibly separate the groups.
  Higher or lower: more separation is better.
  Common pitfall: two distributions can look separated on a small demo set and still fail on real data.
- EER
  What it is: operating point where false accepts and false rejects are equal.
  Why it matters: widely used scalar summary in verification work.
  Higher or lower: lower is better.
  Common pitfall: EER does not tell you whether that operating point is acceptable for the actual use case.
- ROC-AUC
  What it is: threshold-free summary of ranking quality.
  Why it matters: helpful when the chosen threshold is still under discussion.
  Higher or lower: higher is better.
  Common pitfall: a good ROC-AUC can still coexist with poor calibration.
- DET curve
  What it is: false-positive versus false-negative trade-off in verification space.
  Why it matters: emphasizes operating behavior instead of just ranking quality.
  Higher or lower: curves closer to the origin are better.
  Common pitfall: DET curves are easy to over-read on tiny test sets.

## Spoof Metrics
- ROC-AUC and PR-AUC
  What it is: threshold-free summaries of spoof ranking quality.
  Why it matters: they show whether spoofs tend to receive higher spoof scores than bona fide speech.
  Higher or lower: higher is better.
  Common pitfall: PR-AUC is sensitive to class balance and should be read with that context.
- Validation balanced accuracy
  What it is: the average of spoof recall and bona fide recall on the validation split during spoof training.
  Why it matters: it is a safer monitor than raw validation loss or raw accuracy when the spoof branch sees skewed data.
  Higher or lower: higher is better.
  Common pitfall: balanced accuracy is still only a branch-level signal; it does not guarantee stronger final three-way fusion behavior.
- EER
  What it is: point where spoof misses and false spoof alarms balance.
  Why it matters: convenient single-number summary for anti-spoof work.
  Higher or lower: lower is better.
  Common pitfall: a good EER does not guarantee a good operational threshold.
- Confusion matrix
  What it is: count table of true versus predicted classes.
  Why it matters: shows whether errors are mostly missed spoofs or false spoof flags.
  Higher or lower: diagonals should dominate.
  Common pitfall: raw counts are hard to compare across imbalanced classes, so use the normalized view too.

## Joint-System Metrics
- Final decision accuracy
  What it is: correctness of the fused three-way decision.
  Why it matters: this is the closest metric to the actual research question.
  Higher or lower: higher is better.
  Common pitfall: on a spoof-heavy dataset it can look acceptable even when a trivial always-spoof classifier is competitive or better.
- Classwise precision and recall
  What it is: per-class error summaries for `target_bona_fide`, `spoof`, and `wrong_speaker`.
  Why it matters: helps identify whether the system mainly confuses spoof with wrong speaker, or both with target bona fide.
  Higher or lower: higher is better.
  Common pitfall: small classes can make these numbers unstable on demo data.
- Majority-class baseline
  What it is: a trivial baseline that predicts `spoof` for every trial.
  Why it matters: the tracked ASVspoof alpha baseline is heavily skewed toward `spoof`, so this baseline is the minimum honest comparison for any reported joint accuracy.
  Higher or lower: the proposed method should ideally beat it on both macro F1 and balanced accuracy, not just on accuracy.
  Common pitfall: beating the majority baseline on accuracy alone can still hide poor classwise behavior.
- Macro F1
  What it is: unweighted average of the three classwise F1 values for `target_bona_fide`, `spoof`, and `wrong_speaker`.
  Why it matters: it is a stronger headline metric than raw accuracy for the imbalanced joint task.
  Higher or lower: higher is better.
  Common pitfall: it can be noisy on very small datasets, but on real-data alpha baselines it is still usually more honest than accuracy.
- Balanced accuracy
  What it is: average recall across the three final decision classes.
  Why it matters: it shows whether one class is being ignored even when overall accuracy looks acceptable.
  Higher or lower: higher is better.
  Common pitfall: balanced accuracy says nothing about precision, so it should be read with the classwise table.
- Threshold sweep analysis
  What it is: grid evaluation over SV and spoof thresholds.
  Why it matters: shows whether the apparent performance depends on one brittle threshold choice and supports validation-only threshold tuning.
  Higher or lower: broader stable high-value regions are better.
  Common pitfall: threshold sweeps on synthetic data should not be overgeneralized.
- Decision-path summary
  What it is: counts for the main final-decision routes such as the spoof gate, SV accept after spoof rejection, and SV reject after spoof rejection.
  Why it matters: shows whether the spoof branch is dominating the joint decision space or letting too many spoofed trials pass into the SV stage.
  Higher or lower: there is no universal better direction; interpret the path mix together with the true-spoof fraction in each path.
  Common pitfall: a dominant spoof gate can look reassuring while still hiding poor specificity or weak class balance.

## Calibration Metrics
- Brier score
  What it is: squared error of probabilistic predictions.
  Why it matters: punishes both overconfidence and underconfidence.
  Higher or lower: lower is better.
  Common pitfall: it mixes calibration and discrimination into one number.
- Reliability diagram
  What it is: plot of predicted probability against observed frequency.
  Why it matters: the most intuitive way to inspect calibration.
  Higher or lower: closer to the diagonal is better.
  Common pitfall: sparse bins can make the curve noisy.
- Expected calibration error
  What it is: average gap between confidence and observed frequency across bins.
  Why it matters: useful scalar summary when discussing thresholded probability outputs.
  Higher or lower: lower is better.
  Common pitfall: ECE depends on the chosen binning scheme.

## Localization and Explainability Metrics
- Segment precision, recall, and F1
  What it is: frame or segment agreement between predicted suspicious regions and labeled suspicious regions.
  Why it matters: useful when partial-spoof labels exist.
  Higher or lower: higher is better.
  Common pitfall: these metrics depend strongly on the segment definition.
- IoU
  What it is: overlap quality between predicted suspicious spans and labeled spans.
  Why it matters: more interpretable than framewise counts when regions are contiguous.
  Higher or lower: higher is better.
  Common pitfall: can punish slightly misaligned boundaries harshly.
- Faithfulness-style analysis
  What it is: checks whether the explanation aligns with model behavior under perturbation or masking.
  Why it matters: stronger than qualitative explanation screenshots alone.
  Higher or lower: depends on the chosen test, but stronger agreement is better.
  Common pitfall: still does not prove causal truth about the real world.

## Alpha-Review Interpretation Rule
- Use joint metrics first because the repository's main question is about the final enrollment-conditioned decision.
- For the three-way task, prefer macro F1, balanced accuracy, classwise metrics, and majority-baseline comparison before quoting plain accuracy.
- Use SV and spoof metrics second because they explain why the fused system behaves as it does.
- Use calibration and explainability outputs as supporting evidence rather than headline evidence.
