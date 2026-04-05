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

## Quality Scan Modes
Large private corpora can make a full waveform pass expensive during staging. The importer therefore supports three quality scan modes:

- `full`: decode every waveform and compute complete quality statistics
- `header_only`: use audio headers only for duration and sample-rate checks
- `header_plus_sample`: compute header-based summaries for the full corpus and full waveform summaries for a deterministic subset

Trade-off:

- `full` gives the richest review tables but is slowest
- `header_only` is fastest but leaves speech ratio, clipping, RMS, and SNR proxy as missing values
- `header_plus_sample` is the recommended compromise for large private corpora because it preserves fast corpus-wide staging while still giving supervisors some waveform-derived quality evidence

## Import Steps
1. Copy [`configs/private_corpus_template.yaml`](../configs/private_corpus_template.yaml) to a new config file.
2. Set `dataset_root`, `raw_metadata_path`, and `manifest_output_dir`.
3. Decide whether to preserve provided splits or generate speaker-disjoint splits.
4. Choose a `quality_scan_mode`.
5. Set the wrong-speaker trial policy if you want more than one impostor probe per claimed speaker.
6. Run:

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
- Enrollment uses a deterministic seeded ranking that prefers distinct `source_recording_id` values before reusing additional bona fide utterances.
- `target_bona_fide` trials use later bona fide utterances from the same speaker after enrollment utterances are removed from the candidate pool.
- `spoof` trials use spoofed utterances from the claimed speaker when they do not share a `source_recording_id` with enrollment audio.
- `wrong_speaker` trials use bona fide probes from other speakers in the same split.
- `wrong_speaker_trials_per_speaker` controls how many impostor probes each claimed speaker receives when enough leakage-safe candidates exist.
- `impostor_sampling_strategy` controls whether wrong-speaker probes are picked with a round-robin policy across impostor speakers or a seeded global shuffle.

Why this matters:

- weak wrong-speaker coverage can make the final 3-class evaluation look better than it really is
- enrollment selection tied to file order can accidentally bias the evaluation toward a particular recording pattern
- the current staging logic is designed to stay deterministic without inheriting those ordering artifacts

## What Invalidates The Run
The importer fails if:
- required metadata columns are missing
- referenced audio files do not exist
- too few speakers exist to support leakage-safe evaluation
- a speaker appears in multiple splits when speaker-disjoint mode is required
- a trial would reuse the same source recording for enrollment and probe
- the chosen split cannot support all required trial labels
- duration filters remove all usable utterances

## Recommended Review Order
1. Open `dataset_summary.json`.
2. Check `speaker_split_report.csv`.
3. Check `leakage_report.csv`.
4. Inspect `quality_summary.csv`.
5. Only then run the SV, spoof, and fusion experiments.
