# OOF Seed Robustness Ablation

## Overview

This directory contains a methodological ablation examining the effect of random seed variation on the supervised out-of-fold probe evaluation used in the Phase 2 medical QA setting.

In this context, the seed controls stochastic components of the evaluation pipeline, including cross-validation partitioning and related seeded procedures. The purpose of the ablation is to assess whether the reported probe results are stable across reasonable seed choices.

---

## Hypothesis

Out-of-fold probe performance is expected to remain broadly stable under reasonable variation in the random seed. Moderate seed changes should not materially alter the main conclusions of the evaluation.

---

## Motivation / Background

Phase 2 evaluates white-box scores for uncertainty quantification and hallucination or error detection in constrained medical QA tasks. For the supervised probes, evaluation relies on leakage-safe out-of-fold predictions, which depend on seeded data partitioning and other seeded components of the pipeline.

Because these procedures can introduce small run-to-run variation, it is important to verify that reported results do not depend strongly on a particular seed. This ablation therefore serves as a robustness check on whether the observed probe behavior reflects a stable detection signal rather than a seed-specific partitioning effect.

---

## Ablation Design

The ablation varies only the global random seed used in the evaluation pipeline.

Evaluated settings:

- **seed0**
- **seed42**
- **seed123**
- **seed999**
- **seed2026**

The runs in this directory cover the Phase 2 medical QA configurations provided for this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operationally, the ablation is implemented by changing:

- `--seed <value>`

The maintained batch runner for this experiment is:

- `phase_2_medical/scripts/run_ablation_oof_seeds.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including bootstrap budget (`B=5000`), cross-validation split count (`n_splits=5`), hidden-layer selection (`16`), and pooling strategy (`mean_answer`). Each setting directory contains the corresponding run artifacts, including result files and manifests for direct comparison across seeds.

---

## Expected Effect

If the evaluation pipeline is robust, the reported metrics should vary only modestly across the tested seeds. Small fluctuations are expected because fold assignments and related seeded procedures change, but large shifts would indicate undesirable sensitivity to initialization or partitioning.

For uncertainty quantification and hallucination detection, this matters because strong seed dependence would weaken the claim that the reported probe behavior reflects a stable methodological effect.

---

## Notes / Interpretation

This ablation should be interpreted as a sensitivity analysis of procedural robustness. It is not intended as a hyperparameter search over seeds, nor as evidence that a particular seed is intrinsically preferable.

Accordingly, the main quantity of interest is the stability of the evaluation outcome across reasonable seed choices. If substantial variation remains, the associated uncertainty and hallucination-detection results should be interpreted more cautiously.