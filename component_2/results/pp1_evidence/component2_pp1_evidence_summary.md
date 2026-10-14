# Component 2 – PP1 Evidence Summary

## Current implementation
- Smartphone pilot: 896 cleaned reviews across 10 products.
- Function 1: preliminary TF-IDF + Ridge usefulness model with held-out evaluation.
- Semantic representation: Sentence-BERT embeddings for all 896 pilot reviews.
- Function 2: fixed-budget constrained selector selecting 5 reviews per product.
- Baseline, ablation, sensitivity and exploratory statistical validation completed.

## Prototype selector evidence
- Full Proposed mean usefulness: 0.743; Predicted-Usefulness Top-K: 0.859.
- Full Proposed supported-minority recall: 0.417; Predicted-Usefulness Top-K: 0.278.
- Full Proposed priority-minority coverage: 1.00; Proposed without Minority: 0.20.

## Statistically supported prototype ablations
- Full Proposed vs NoMinority on Supported-minority recall: benefit 0.136, 95% CI [0.082, 0.195], Holm p=0.0117.
- Full Proposed vs NoMinority on Priority-minority coverage: benefit 0.800, 95% CI [0.650, 0.950], Holm p=0.0098.
- Full Proposed vs NoRepresentativeness on Semantic representativeness: benefit 0.119, 95% CI [0.082, 0.155], Holm p=0.0098.

## Claim boundary
- C1-derived aspect/sentiment labels are synthetic integration-test placeholders.
- Minority, coverage and opinion-fidelity values are prototype diagnostics, not final research findings.
- Statistical tests are exploratory because the current pilot contains only 10 products.
- Real C1 integration, broader category evaluation and final model/parameter validation remain future work.