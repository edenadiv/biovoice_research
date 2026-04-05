# Latest Run Review Artifacts

This directory contains a small Git-tracked review bundle copied from the source run:

- `outputs/runs/20260405_200907_fusion_plus_interpretable_features`

That source run is the latest complete real-data baseline run currently available in the repository. It uses the official ASVspoof LA pipeline and should be treated as real-data alpha-baseline evidence, not as a production claim.

## Why This Folder Exists

The full experiment storage under `outputs/` is intentionally ignored because it contains large and frequently changing artifacts such as checkpoints, complete plot directories, notebook exports, and full prediction tables. This folder keeps only the small text-heavy and tabular artifacts that supervisors can inspect directly on GitHub.

## Included Files

- `supervisor_report.md`: high-level review narrative for supervisors
- `metrics.json`: structured metrics from the selected run
- `dataset_review.json`: dataset mode, split assumptions, and leakage summary
- `metric_summary.csv`: flattened metric table for quick spreadsheet-style review
- `plot_inventory.md`: interpretation notes for the saved figures in the source run
- `predictions_head.csv`: first 200 prediction rows from the full prediction table for schema and spot-check review
- `metadata.json`: provenance for this curated artifact bundle

## Important Notes

- These are review artifacts, not full experiment storage.
- Large artifacts such as checkpoints, complete plot directories, notebook exports, and the full `predictions.csv` remain in the local ignored `outputs/` tree.
- The original prediction table was not committed because it is large (`162,992` rows, about `53 MB`). This folder therefore includes `predictions_head.csv` instead of the full file.
- The copied supervisor-facing files were lightly normalized for GitHub readability:
  - machine-local absolute paths were replaced with repository-relative paths where appropriate
  - one stale demo-data interpretation note was corrected to reflect real-data baseline evidence

## Missing Or Omitted Files

- No required review artifact was missing in the selected source run.
- The full `predictions.csv` was intentionally omitted from Git tracking because of size; `predictions_head.csv` is included instead and documented here.
