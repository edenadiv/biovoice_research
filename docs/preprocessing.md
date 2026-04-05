# Preprocessing

## Implemented Steps
- Sample-rate normalization
- Mono conversion
- RMS loudness normalization
- Simple silence trimming
- Padding and truncation to a common duration
- Overlapping segmentation for segment-level analysis

## Why These Steps Exist
- Keep input distributions stable across runs
- Reduce trivial channel and duration variation
- Enable consistent batching for compact baselines
- Expose segment-level evidence for suspiciousness plots

## Quality Checks
The preprocessing layer also records duration, speech ratio, clipping ratio, peak amplitude, RMS, and an SNR proxy. These values support quality plots, robustness breakdowns, and reviewer inspection.
