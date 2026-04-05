# Data Protocol

## Manifest Types
- `utterances.csv`: utterance-level training examples
- `trials.csv`: enrollment-conditioned evaluation trials

## Raw Private-Corpus Metadata
The real-data baseline expects one user-managed metadata table with:
- `utterance_id`
- `speaker_id`
- either `path` or `relative_path`
- `spoof_label`
- `source_recording_id`

Optional columns:
- `split`: if present and `use_existing_splits=true`, the importer preserves the provided split labels

The importer stages this raw table into canonical `utterances.csv` and `trials.csv`, plus split manifests and review artifacts.

## Leakage Prevention and Split Protocol
- Prefer speaker-disjoint train, validation, and test splits when the protocol permits it.
- Never allow enrollment and probe to come from the same source recording within a trial.
- Never allow overlapping segments from one source file to appear in both enrollment and probe roles.
- Keep multiple utterances per speaker grouped carefully during split generation so unsupported leakage does not occur.
- Save split manifests as first-class artifacts for every run.
- Fail the staging step if speaker-disjoint constraints or trial-level source overlap constraints are violated.

## Synthetic/Demo Data Assumptions
- Synthetic data is used only for smoke testing and artifact validation.
- Synthetic labels demonstrate pipeline semantics, not real-world performance.
- Synthetic generation assumptions are saved alongside the demo manifests.

## Staged Review Artifacts
The staged manifest directory stores:
- `utterances.csv`
- `trials.csv`
- `splits/`
- `dataset_summary.json`
- `quality_summary.csv`
- `leakage_report.csv`
- `speaker_split_report.csv`
