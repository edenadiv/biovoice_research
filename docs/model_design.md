# Model Design

## Speaker Verification Baseline
The SV branch uses a compact log-Mel convolutional encoder. During inference, enrollment embeddings are averaged into a speaker template and compared with the probe embedding using cosine similarity.

## Anti-Spoof Baseline
The spoof branch uses a compact CNN on log-Mel features and outputs an utterance-level spoof probability. The same model is reused on probe segments to localize suspicious regions.

## Fusion
The default alpha fusion is late fusion. It combines speaker similarity, spoof probability, and interpretable enrollment-probe mismatch magnitude. A shallow trainable fusion head is included for future expansion.

## Explainability
Explainability combines:
- segment-level spoof probability
- segment-level speaker similarity
- interpretable feature mismatches
- short generated reasoning statements

These outputs support inspection. They do not prove causality.
