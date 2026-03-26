# Cross-Validation Splits Ablation

## Overview

This directory contains a methodological ablation examining the effect of the cross-validation split count on the supervised probe evaluation used in the Phase 2 medical QA setting.

In this context, `n_splits` denotes the number of folds used to generate leakage-safe out-of-fold probe predictions. The purpose of the ablation is to assess whether probe-level results are stable across reasonable choices of cross-validation granularity.

---

## Hypothesis

Probe performance is expected to remain broadly stable across reasonable values of `n_splits`. Moderate changes in fold count should not materially alter the main conclusions of the evaluation.

---

## Motivation / Background

Phase 2 evaluates white-box scores for uncertainty quantification and hallucination or error detection in constrained medical QA tasks. For the supervised probes, performance is estimated from out-of-fold predictions in order to avoid evaluation leakage.

The number of cross-validation folds affects how the data are partitioned between training and held-out evaluation across the probe pipeline. As a result, it can influence both the amount of training data available per fold and the variability introduced by the partitioning scheme. This ablation therefore serves as a robustness check on whether the reported probe results depend strongly on an arbitrary cross-validation choice.

---

## Ablation Design

The ablation varies only the number of cross-validation splits used for out-of-fold probe evaluation.

Evaluated settings:

- **K3**
- **K5**
- **K10**

The runs in this directory cover the Phase 2 medical QA configurations provided for this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operationally, the ablation is implemented by changing:

- `--n_splits <value>`

The maintained batch runner for this experiment is:

- `phase_2_medical/scripts/run_ablation_n_splits.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including hidden-layer selection (`16`), pooling strategy (`mean_answer`), bootstrap budget (`B=5000`), and random seed (`42`). Each setting directory contains the corresponding run artifacts, including result files and manifests for direct comparison across fold counts.

---

## Expected Effect

If the supervised evaluation pipeline is robust, the reported probe metrics should vary only modestly across the tested values of `n_splits`. Small fluctuations are expected because fold composition changes, but large shifts would indicate sensitivity to the partitioning scheme.

For uncertainty quantification and hallucination detection, this matters because strong dependence on cross-validation granularity would weaken the claim that the observed probe behavior reflects a stable detection signal rather than an artefact of a particular evaluation protocol.

---

## Notes / Interpretation

This ablation should be interpreted as a sensitivity analysis of the supervised evaluation procedure. It is not intended to identify an optimal fold count in isolation, but to test whether the reported results are robust to reasonable choices of cross-validation granularity.

Accordingly, the main quantity of interest is the stability of the evaluation outcome across `n_splits`, rather than any isolated improvement associated with a single fold configuration.