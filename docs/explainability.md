# Explainability

## What the Repository Explains
- Which segments look suspicious
- Whether the probe stays biometrically consistent with enrollment
- Which interpretable features diverge the most
- Short natural-language cues summarizing the evidence

## Why Segment-Level Analysis Matters
- Spoof artifacts may be localized rather than global
- Segment timelines help reviewers inspect whether suspicion clusters in meaningful regions
- Speaker similarity can drift over time even when the overall utterance score looks acceptable

## Important Limitation
Explainability outputs are supporting evidence. They are not proof that a specific acoustic property caused the model decision.
