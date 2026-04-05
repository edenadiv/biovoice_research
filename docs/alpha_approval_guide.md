# Alpha Approval Guide

## Alpha Exit Criteria
- The repository runs end-to-end on synthetic/demo data.
- One baseline SV run completes.
- One baseline spoof run completes.
- One fusion evaluation run completes.
- Mandatory alpha figures are saved.
- The supervisor notebook opens cleanly from saved artifacts.
- Documentation is complete enough for review.
- No fabricated results or unsupported claims appear in the repo.

## What Alpha Approval Means
Alpha approval means the methodology, artifact structure, and experimental plumbing are coherent enough for broader evaluation. It does not mean the system is deployment-ready or scientifically complete.

For real-data baselines, alpha approval means:

- the dataset import path is leakage-safe and reviewable
- the wrong-speaker and spoof trial policies are explicit
- the resulting metrics are treated as baseline evidence, not as strong final claims

## Evidence Package For Review
- Config snapshots
- Split manifests
- Metrics JSON
- Prediction tables
- Mandatory figures
- Run report
- Supervisor report
- Supervisor walkthrough notebook
- Artifact index and plot inventory

## Suggested Supervisor Review Sequence
1. Read the supervisor report.
2. Check the alpha exit checklist JSON.
3. Open the plot inventory so every figure has an interpretation note.
4. Review the confusion matrices, score distributions, and threshold heatmap.
5. Review the explainability case study only after the global figures are understood.

## Real-Data Caution
When the run uses a private corpus or ASVspoof:

- treat measured numbers as evidence for that dataset and protocol only
- give more weight to macro F1, balanced accuracy, and classwise confusion than to raw accuracy alone
- inspect the dataset review section before discussing the model, because split or trial policy can change the meaning of the final metrics
