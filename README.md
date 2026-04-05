# Explainable Spoof-Aware Speaker Verification for Audio Deepfake Detection

Research-first alpha repository for enrollment-conditioned, explainable spoof-aware speaker verification. The repository is designed for methodology approval, experiment execution, alpha evaluation, supervisor review, and future paper writing. It is intentionally not an app or deployment stack.

## Research Question
How can we build an enrollment-based, explainable, spoof-aware speaker verification system that detects audio deepfakes and explains its decisions using biometric consistency and acoustic evidence?

## Alpha Scope
- Dataset-agnostic pipeline with ASVspoof-style trial conventions
- Four experiment modes: `sv_only`, `spoof_only`, `fusion`, `fusion_plus_interpretable_features`
- Self-contained PyTorch and torchaudio baselines
- Plot-heavy experimental outputs
- Supervisor-facing reports and notebooks
- Reproducible synthetic/demo path for smoke tests

## What This Repository Is For
- methodology approval
- experiment execution
- alpha evaluation
- supervisor inspection
- future paper writing

## What This Repository Is Not For
- deployment
- serving APIs
- frontend work
- enterprise orchestration
- product UX

## Why This Repository Uses Enrollment
- Enrollment audio gives the system a claimed-speaker reference, which makes wrong-speaker detection possible.
- Speaker verification alone is insufficient because a spoofed voice can still mimic the target speaker.
- Spoof detection alone is insufficient because a bona fide wrong speaker may not look spoofed.
- Fusion is needed to separate `target_bona_fide`, `spoof`, and `wrong_speaker`.
- Explainability and segment-level analysis are needed so supervisors can inspect where the system finds evidence rather than trusting a single scalar score.

## Repository Structure
```text
configs/      YAML experiment settings
docs/         Methodology, metrics, plotting, and alpha-review documentation
notebooks/    Supervisor-facing walkthrough notebooks
scripts/      Lightweight experiment entry points
src/biovoice/ Research package implementation
tests/        Unit and smoke tests
outputs/      Saved runs, figures, tables, reports, and notebook exports
```

For real-data setup details, start with [Real-Data Onboarding](docs/real_data_onboarding.md).

## Installation
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## First Commands To Run
```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train_baseline_sv.py --config configs/default.yaml
python scripts/train_baseline_spoof.py --config configs/default.yaml
python scripts/train_joint_model.py --config configs/default.yaml
python scripts/generate_supervisor_report.py --config configs/default.yaml
pytest
```

## Real-Data Baseline Commands
```bash
cp configs/private_corpus_template.yaml configs/my_private_corpus.yaml
# edit dataset_root, raw_metadata_path, and manifest_output_dir
python scripts/import_real_dataset.py --config configs/my_private_corpus.yaml
python scripts/train_baseline_sv.py --config configs/my_private_corpus.yaml
python scripts/train_baseline_spoof.py --config configs/my_private_corpus.yaml
python scripts/train_joint_model.py --config configs/my_private_corpus.yaml
```

## Official ASVspoof 2021 LA Commands
```bash
# first place the official archives under external_data/asvspoof/raw/
python scripts/import_asvspoof2021_la.py --config configs/asvspoof2021_la.yaml
python scripts/train_baseline_sv.py --config configs/asvspoof2021_la.yaml
python scripts/train_baseline_spoof.py --config configs/asvspoof2021_la.yaml
python scripts/train_joint_model.py --config configs/asvspoof2021_la.yaml
```

## Data Format
The repo uses two manifest types:
- `utterances.csv`: utterance-level training data with `utterance_id`, `speaker_id`, `path`, `split`, `spoof_label`
- `trials.csv`: enrollment-conditioned trials with `trial_id`, `speaker_id`, `probe_path`, `enrollment_paths`, `label`, `split`

Synthetic/demo data is generated under `demo_data/` and is explicitly labeled as smoke-test only.

For real private-corpus staging, the raw metadata table must include:
- `utterance_id`
- `speaker_id`
- either `path` or `relative_path`
- `spoof_label`
- `source_recording_id`

Optional raw metadata columns:
- `split`: use this if you want to keep a pre-defined split protocol

If `split` is omitted, the importer generates a deterministic speaker-disjoint split when possible.

For a fuller explanation of leakage rules and manifest semantics, see [Data Protocol](docs/data_protocol.md).

## Key Entry Points
- CLI: `biovoice`
- Data preparation: [`scripts/prepare_data.py`](scripts/prepare_data.py)
- Real-data import: [`scripts/import_real_dataset.py`](scripts/import_real_dataset.py)
- Official ASVspoof import: [`scripts/import_asvspoof2021_la.py`](scripts/import_asvspoof2021_la.py)
- SV baseline: [`scripts/train_baseline_sv.py`](scripts/train_baseline_sv.py)
- Spoof baseline: [`scripts/train_baseline_spoof.py`](scripts/train_baseline_spoof.py)
- Joint fusion run: [`scripts/train_joint_model.py`](scripts/train_joint_model.py)
- Supervisor report: [`scripts/generate_supervisor_report.py`](scripts/generate_supervisor_report.py)

## Output Artifacts
Each run saves:
- logs
- config snapshots
- checkpoints
- `metrics.json`
- predictions tables
- plots
- reports
- explainability case files
- calibration outputs
- ablation summary tables and figures

The polished run layout now also includes:
- `tables/artifact_index.csv`: full file inventory for the run
- `reports/artifact_index.md`: supervisor-friendly artifact list
- `tables/metric_summary.csv`: flattened metric table
- `reports/plot_inventory.md`: per-figure interpretation notes
- `tables/threshold_sweep.csv`: operating-point sweep table
- `reports/dataset_review.json`: dataset assumptions, split protocol, and leakage summary for the run

Runs live under `outputs/runs/<timestamp>_<experiment>/`.

## Latest Review Artifacts
For GitHub review, the repository tracks a small curated artifact bundle under [artifacts/latest_run](artifacts/latest_run/). This folder contains the latest supervisor-facing text and tabular evidence from the most recent meaningful run, while the full local run bundle still lives under the ignored `outputs/` tree.

Use the tracked artifact bundle when you want a quick GitHub-readable view of the latest baseline evidence. Use `outputs/runs/<timestamp>_<experiment>/` locally when you need the full plots, checkpoints, notebooks, and complete prediction tables.

## How Supervisors Should Read A Run
1. Open `reports/supervisor_report.md` first for the high-level verdict.
2. Open `reports/plot_inventory.md` next so each figure is paired with an interpretation note.
3. Review `metrics.json` and `tables/metric_summary.csv` together.
4. Inspect `predictions.csv` and the explainability case files only after the global figures look sensible.
5. Treat demo-data numbers as pipeline evidence and real-data numbers as alpha baseline evidence for the evaluated protocol only.

## Alpha Exit Criteria
- Repository runs end-to-end on synthetic/demo data
- One SV baseline run completes
- One spoof baseline run completes
- One fusion evaluation run completes
- Mandatory alpha figures are saved
- Supervisor notebook opens and reads saved artifacts cleanly
- Documentation is complete enough for supervisor review
- No fabricated results or unsupported claims are presented

## Mandatory Alpha Figures
- Class balance plot
- Duration histogram
- Train/validation loss curves
- ROC, PR, and DET curves where applicable
- Confusion matrix
- Normalized confusion matrix
- Target vs non-target score distributions
- Spoof vs bona fide score distributions
- Reliability diagram
- Threshold sweep heatmap
- SV score vs spoof probability scatter plot
- Ablation summary bar chart
- Waveform with suspicious segments
- Spoof score over time
- Speaker similarity over time

## Limitations
- These baselines are research baselines, not production systems.
- Synthetic/demo data only validates plumbing, not real-world performance.
- Explainability outputs are supporting evidence, not proof of causality.
- Probability outputs should be interpreted with the reliability diagram rather than assumed to be perfectly calibrated.
- Wrong-speaker decision quality depends strongly on how impostor probes are constructed, so real-data staging policy matters.
- Stronger pretrained backbones and broader benchmarking are future work.

## Next Steps After Alpha
- Replace compact baselines with stronger pretrained backbones
- Add broader ablation and robustness coverage
- Benchmark on larger real datasets
- Expand explainability and calibration analysis
