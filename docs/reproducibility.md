# Reproducibility

## Implemented Controls
- Global deterministic seed utilities
- Saved YAML config snapshots
- Saved split manifests
- Saved checkpoints
- Saved predictions and metrics
- Stable output directory structure
- Deterministic enrollment selection
- Deterministic wrong-speaker probe selection
- Saved dataset review metadata for real-data runs

## Practical Notes
- Use a clean Python 3.11 or 3.12 virtual environment.
- Re-run with the same config snapshot when comparing results.
- Treat synthetic/demo results as smoke-test artifacts, not benchmark evidence.
- For private corpora, keep the same `quality_scan_mode`, `quality_waveform_sample_size`, and seed if you want staging summaries to be comparable across runs.
- If you change `wrong_speaker_trials_per_speaker` or `impostor_sampling_strategy`, treat the resulting metrics as a different evaluation protocol.
