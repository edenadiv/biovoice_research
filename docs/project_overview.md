# Project Overview

This repository studies enrollment-conditioned audio deepfake detection as a joint decision problem rather than a single-task classification problem.

## Problem Framing
- Input: multiple enrollment utterances from a claimed speaker plus one probe utterance
- Outputs: spoof probability, speaker-match score, final three-way decision, suspicious segments, and explanation cues
- Goal: detect deepfakes while also distinguishing them from bona fide wrong-speaker trials

## Why Joint Modeling Matters
- Speaker verification alone can accept convincing spoofs.
- Spoof detection alone cannot reject bona fide wrong speakers.
- Enrollment lets the system compare probe behavior against a speaker-specific reference.
- Explainability helps reviewers inspect whether the decision is grounded in biometric consistency and acoustic mismatch evidence.

## Alpha Deliverable
The alpha package is successful when supervisors can inspect code, config, artifacts, plots, reports, and notebooks and conclude that the methodology is coherent enough for broader evaluation.
