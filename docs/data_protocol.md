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

## Real-Data Split And Trial Policy
The private-corpus importer is deterministic, but it is no longer tied to raw file ordering.

Enrollment policy:

- choose `enrollment_count` bona fide utterances per claimed speaker
- prefer distinct `source_recording_id` values before reusing additional recordings
- apply a seeded ranking so two runs with the same seed are reproducible

Target and spoof probe policy:

- target probes are bona fide utterances from the claimed speaker that were not used for enrollment
- spoof probes are spoofed utterances from the claimed speaker
- both are filtered to avoid enrollment/probe source overlap

Wrong-speaker policy:

- wrong-speaker probes are bona fide utterances from other speakers in the same split
- `wrong_speaker_trials_per_speaker` controls multiplicity
- `impostor_sampling_strategy` controls whether the importer cycles across impostor speakers or uses a seeded shuffled pool
- the goal is to create a stronger impostor set without violating split or source-recording safety

## Quality Scan Modes
The private-corpus path supports:

- `full`
- `header_only`
- `header_plus_sample`

Interpretation:

- `full` is the most complete and the most expensive
- `header_only` is fast and sufficient for duration filtering, but not for full quality analysis
- `header_plus_sample` is the best option when the corpus is large enough that decoding every waveform would slow staging substantially

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

For private-corpus runs, `dataset_summary.json` also records:

- quality scan mode
- counts of waveform-scanned vs header-only files
- enrollment policy summary
- wrong-speaker construction summary
