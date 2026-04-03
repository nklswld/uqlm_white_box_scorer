# Cross-Validation Splits Ablation

## Aim

This ablation varies `n_splits`, the number of folds used to generate leakage-safe out-of-fold probe predictions in Phase 2. It tests whether the supervised readouts depend materially on fold granularity.

---

## Settings

Only the fold count is changed.

Evaluated settings:

- **K3**
- **K5**
- **K10**

The released runs cover the available Phase 2 medical QA configurations in this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operational change:

- `--n_splits <value>`

Runner:

- `phase_2_medical/scripts/run_ablation_n_splits.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including hidden-layer selection (`16`), pooling strategy (`mean_answer`), bootstrap budget (`B=5000`), and random seed (`42`). Each setting directory contains the corresponding results and manifests.

---

## Why It Matters

Changing `n_splits` changes both the train/held-out proportion per fold and the partitioning pattern seen by the supervised probes. If the probe results move substantially across `K3`, `K5`, and `K10`, the reported behavior is sensitive to the evaluation design rather than just to the underlying signal.

Small shifts are expected; large shifts would call for more caution when interpreting supervised probe results.

---

## Reading the Results

This ablation is a sensitivity check on the supervised evaluation procedure. The central question is whether the out-of-fold probe results remain stable across reasonable fold counts, not which single `n_splits` value appears best in isolation.
