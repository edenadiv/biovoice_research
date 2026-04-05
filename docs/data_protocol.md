# Data Protocol

## Manifest Types
- `utterances.csv`: utterance-level training examples
- `trials.csv`: enrollment-conditioned evaluation trials

## Leakage Prevention and Split Protocol
- Prefer speaker-disjoint train, validation, and test splits when the protocol permits it.
- Never allow enrollment and probe to come from the same source recording within a trial.
- Never allow overlapping segments from one source file to appear in both enrollment and probe roles.
- Keep multiple utterances per speaker grouped carefully during split generation so unsupported leakage does not occur.
- Save split manifests as first-class artifacts for every run.

## Synthetic/Demo Data Assumptions
- Synthetic data is used only for smoke testing and artifact validation.
- Synthetic labels demonstrate pipeline semantics, not real-world performance.
- Synthetic generation assumptions are saved alongside the demo manifests.
