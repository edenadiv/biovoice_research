# Real-Data Onboarding

## Purpose
This document explains how to import one private corpus into the existing BioVoice research pipeline without changing the canonical manifest schema.

## Expected Raw Inputs
You need:
- a directory of local audio files
- one metadata table in CSV or JSON format

Required metadata columns:
- `utterance_id`
- `speaker_id`
- either `path` or `relative_path`
- `spoof_label`
- `source_recording_id`

Optional metadata columns:
- `split`

If `path` is used, values may be absolute or repository-relative. If `relative_path` is used, values are resolved relative to `dataset_root`.

## Import Steps
1. Copy [private_corpus_template.yaml](/Users/edenadiv/Desktop/biovoice/configs/private_corpus_template.yaml) to a new config file.
2. Set `dataset_root`, `raw_metadata_path`, and `manifest_output_dir`.
3. Decide whether to preserve provided splits or generate speaker-disjoint splits.
4. Run:

```bash
python scripts/import_real_dataset.py --config configs/my_private_corpus.yaml
```

## What The Importer Produces
The importer writes:
- canonical `utterances.csv`
- canonical `trials.csv`
- split manifests under `splits/`
- `dataset_summary.json`
- `quality_summary.csv`
- `leakage_report.csv`
- `speaker_split_report.csv`

These staged files become the only inputs used by the training and evaluation scripts.

## Trial Construction Rules
- Enrollment uses the first `enrollment_count` bona fide utterances for each speaker within a split.
- `target_bona_fide` trials use later bona fide utterances from the same speaker.
- `spoof` trials use spoofed utterances from the claimed speaker when they do not share a `source_recording_id` with enrollment audio.
- `wrong_speaker` trials use bona fide probes from another speaker in the same split.

## What Invalidates The Run
The importer fails if:
- required metadata columns are missing
- referenced audio files do not exist
- too few speakers exist to support leakage-safe evaluation
- a speaker appears in multiple splits when speaker-disjoint mode is required
- a trial would reuse the same source recording for enrollment and probe
- duration filters remove all usable utterances

## Recommended Review Order
1. Open `dataset_summary.json`.
2. Check `speaker_split_report.csv`.
3. Check `leakage_report.csv`.
4. Inspect `quality_summary.csv`.
5. Only then run the SV, spoof, and fusion experiments.
