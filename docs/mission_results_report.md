# Mission Results Report

## Scope
This report summarizes the measured outcome of the BioVoice alpha mission:

- build a research-first repository for enrollment-conditioned, explainable, spoof-aware speaker verification
- move from demo-only validation to real-data evaluation
- produce supervisor-readable artifacts, plots, reports, and notebooks
- keep the package suitable for methodology approval rather than deployment

The report focuses on what was actually completed and what the current evidence supports. It does not make claims beyond the measured runs.

## Executive Summary
The mission succeeded at the research-alpha level.

We now have:

- a complete research repository with documented data, modeling, evaluation, explainability, visualization, reporting, and testing layers
- a leakage-checked real-data protocol for ASVspoof LA
- completed end-to-end baseline experiments on official real data
- exported notebooks for supervisor review
- saved figures, predictions, checkpoints, config snapshots, and explanation case files

The strongest evidence bundle is the real fusion run produced on April 5, 2026:

- local artifact bundle: `outputs/runs/20260405_200907_fusion_plus_interpretable_features`

This run is the current reference point for supervisor review.

## Mission Objectives and Status
| objective | status | evidence |
| --- | --- | --- |
| Build research-first repo structure | completed | repository modules, configs, docs, tests, notebooks |
| Support demo and real-data staging | completed | `scripts/prepare_data.py`, `scripts/import_real_dataset.py`, `scripts/import_asvspoof2021_la.py` |
| Implement SV baseline | completed | saved checkpoint and training history in run artifacts |
| Implement spoof baseline | completed | saved checkpoint and training history in run artifacts |
| Implement late-fusion baseline | completed | real fusion run with predictions and metrics |
| Add explainability outputs | completed | segment timelines, suspicious spans, case JSON/CSV files |
| Generate mandatory alpha figures | completed | saved under `plots/` in the fusion run |
| Produce supervisor-readable notebooks | completed | eight exported notebooks under `outputs/notebooks_exports/` |
| Run on official real dataset | completed | ASVspoof 2019/2021 LA staging and evaluation |

## Real Dataset Protocol
The real-data milestone used official ASVspoof LA data in a way that preserves the repo's enrollment-conditioned framing.

Protocol summary:

- training and development data from ASVspoof 2019 LA
- enrollment pool from bona fide ASVspoof 2019 LA evaluation recordings
- probes from ASVspoof 2021 LA evaluation recordings
- enrollment count per claim: `3`
- final evaluation trials: `162,992`

Measured dataset summary from the saved run:

- dataset mode: `asvspoof2021_la`
- dataset name: `ASVspoof2021_LA`
- number of utterances staged: `121,461`
- number of speakers: `107`
- speakers per split:
  - enroll: `67`
  - train: `20`
  - val: `20`
- trials per split:
  - test: `162,992`
- class counts:
  - spoof: `133,360`
  - target_bona_fide: `14,816`
  - wrong_speaker: `14,816`
- mean duration: `3.25 s`
- mean speech ratio: `0.483`

## Leakage and Protocol Integrity
Leakage prevention was a central requirement because the system is enrollment-conditioned.

Verified outcomes from the saved dataset review:

- speaker-disjoint violations: `0`
- trial leakage violations: `0`

Interpretation:

- no unsupported overlap was found between protected speaker groups used by the protocol
- no unsafe enrollment-versus-probe source overlap survived the staging checks
- the real run can therefore be discussed as a valid baseline experiment rather than a compromised data-pipeline demonstration

## Core Measured Results
The following values come from the saved real run metrics and should be treated as the authoritative measured results for the current baseline package.

### Speaker Verification Branch
- accuracy: `0.7927`
- precision: `0.9536`
- recall: `0.8115`
- F1: `0.8768`
- ROC-AUC: `0.7310`
- PR-AUC: `0.9476`
- EER: `0.3265`

Interpretation:

- the SV branch provides meaningful biometric separation
- target trials score substantially higher than non-target trials on average
- the EER is still far too high for a strong verification system, which is expected for a compact research baseline

### Spoof Detection Branch
- accuracy: `0.7887`
- precision: `0.8443`
- recall: `0.9094`
- F1: `0.8757`
- ROC-AUC: `0.6487`
- PR-AUC: `0.8760`
- EER: `0.3815`

Interpretation:

- the spoof branch contributes useful ranking signal
- spoof recall is high, but global ranking quality remains moderate
- this branch helps the fusion system, but it is not yet a strong standalone anti-spoof model

### Final Joint System
- accuracy: `0.7746`
- macro F1: `0.4283`
- weighted F1: `0.7537`
- balanced accuracy: `0.4149`

Interpretation:

- the fused system is functional and measurable on real data
- overall accuracy is helped by the heavy spoof class proportion
- macro F1 and balanced accuracy show that minority-class behavior remains a major weakness
- the present system is better at broad triage than at precise classwise separation

### Calibration
- Brier score: `0.1614`
- expected calibration error: `0.1032`

Interpretation:

- probabilities are usable enough for inspection
- calibration is not yet strong enough to justify aggressive confidence-based claims
- validation-time threshold and calibration work remains a clear next step

## Decision Distribution
The saved prediction table contains `162,992` real trials.

Ground-truth distribution:

- spoof: `133,360`
- target_bona_fide: `14,816`
- wrong_speaker: `14,816`

System final-decision distribution:

- spoof: `143,641`
- target_bona_fide: `12,483`
- wrong_speaker: `6,868`

Interpretation:

- the baseline leans toward declaring `spoof`
- that bias may be partly reasonable given the evaluation mix, but it also suggests under-allocation to the `wrong_speaker` decision
- this is consistent with the relatively weak macro F1

## What Worked Well
- The repo architecture held up under real-data use without needing a redesign.
- Canonical manifests, staging, and validation were sufficient to support official-data experiments.
- The SV and spoof branches both contributed measurable signal.
- The fusion pathway produced a stable, reviewable final decision process.
- Explainability outputs were generated at segment level, not just utterance level.
- Mandatory plots and supervisor materials were created successfully.
- The notebook review flow remained usable after switching from demo data to real data.

## What Was Hard or Fragile
- Real-data onboarding required careful trial construction to avoid enrollment/probe overlap.
- Apple GPU acceleration is still blocked by upstream PyTorch binary support on macOS 26, so the current evidence package was not produced on MPS.
- The class imbalance in ASVspoof LA makes headline accuracy much easier to over-read than macro F1.
- The current baselines remain compact and intentionally simple, so they expose the method but do not represent the strongest possible model family.

## Important Artifacts
Local artifact bundle for the main real run:

- `outputs/runs/20260405_200907_fusion_plus_interpretable_features`

Important files inside that bundle:

- `metrics.json`
- `predictions.csv`
- `reports/supervisor_report.md`
- `reports/run_report.md`
- `reports/dataset_review.json`
- `reports/plot_inventory.md`
- `reports/artifact_index.md`
- `reports/alpha_exit_checklist.json`
- `checkpoints/speaker_model.pt`
- `checkpoints/spoof_model.pt`
- `plots/`
- `explainability/`

Exported notebooks:

- `outputs/notebooks_exports/01_problem_and_data_overview.ipynb`
- `outputs/notebooks_exports/02_preprocessing_and_quality_checks.ipynb`
- `outputs/notebooks_exports/03_baseline_models.ipynb`
- `outputs/notebooks_exports/04_joint_model_training_review.ipynb`
- `outputs/notebooks_exports/05_evaluation_dashboard.ipynb`
- `outputs/notebooks_exports/06_explainability_case_studies.ipynb`
- `outputs/notebooks_exports/07_ablation_study.ipynb`
- `outputs/notebooks_exports/08_supervisor_walkthrough.ipynb`

## Mandatory Figure Coverage
The mission required a minimum set of alpha-review figures. The real fusion run produced the required figure family, including:

- class balance
- duration histogram
- train and validation loss curves
- ROC curves
- PR curves
- DET curves
- confusion matrix
- target versus non-target score distributions
- spoof versus bona fide score distributions
- reliability diagram
- ablation summary bar chart
- waveform with suspicious segments
- spoof score over time
- speaker similarity over time

The interpretation notes for each figure are recorded in the saved plot inventory.

## Alpha Exit Criteria Assessment
The alpha evidence bundle meets the intended research-alpha bar:

- repo runs end-to-end on non-demo data
- baseline SV run completed
- baseline spoof run completed
- fusion evaluation completed
- mandatory figures were generated
- supervisor notebook exports were produced
- documentation and reports are present
- no fabricated results were used

In that sense, the mission is alpha-complete.

## Risks and Limitations
- These are research baselines, not production systems.
- The current results do not justify broader real-world robustness claims.
- Explainability outputs are supporting diagnostic evidence, not proof of causality.
- The final decision layer still struggles with minority-class separation.
- Stronger pretrained backbones and larger-scale benchmarking remain future work.

## Recommendation
The package is strong enough for:

- supervisor review
- methodology discussion
- threshold and calibration refinement
- structured ablation work
- paper-planning preparation

It is not yet strong enough for:

- production positioning
- broad deployment claims
- claims of robust generalization beyond the present benchmark setting

## Next Research Steps
1. Improve threshold selection and validation-time calibration.
2. Run systematic ablations that remove SV, spoof, and interpretable-feature inputs one at a time.
3. Add robustness breakdowns by duration, quality, and enrollment count.
4. Strengthen the spoof branch with a better backbone while keeping the current reporting stack unchanged.
5. Revisit the final decision policy for better separation between `target_bona_fide` and `wrong_speaker`.

## Integrity Statement
This report reflects measured repository and experiment outcomes available in the current workspace. It does not include fabricated numbers, invented baselines, or unsupported scientific claims.
