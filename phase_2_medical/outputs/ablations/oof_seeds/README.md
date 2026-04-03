# OOF Seed Robustness Ablation

## Aim

This ablation varies the global random seed used in the supervised out-of-fold probe evaluation for Phase 2. The seed affects fold assignments and other seeded procedures in the probe pipeline, so the purpose here is to check whether the reported probe behavior is stable across reasonable seed choices.

---

## Settings

Only the global seed is changed.

Evaluated settings:

- **seed0**
- **seed42**
- **seed123**
- **seed999**
- **seed2026**

The released runs cover the available Phase 2 medical QA configurations in this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operational change:

- `--seed <value>`

Runner:

- `phase_2_medical/scripts/run_ablation_oof_seeds.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including bootstrap budget (`B=5000`), cross-validation split count (`n_splits=5`), hidden-layer selection (`16`), and pooling strategy (`mean_answer`). Each setting directory contains the corresponding results and manifests.

---

## Why It Matters

If the evaluation pipeline is robust, the reported metrics should vary only modestly across the tested seeds. Small fluctuations are expected because fold assignments change. Large shifts would indicate undesirable sensitivity to seeded partitioning.

This is therefore a check on procedural robustness rather than a search for a favorable seed.

---

## Reading the Results

The key quantity is dispersion across reasonable seed choices. If substantial variation remains, the associated supervised-probe results should be interpreted more cautiously. A seed with the best isolated metric should not be treated as intrinsically preferable.
