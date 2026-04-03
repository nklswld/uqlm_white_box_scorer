# Hidden Pooling Ablation

## Aim

This ablation varies the pooling rule used to convert token-level hidden states into a single example-level representation for the hidden probe in Phase 2. It tests whether the hidden-probe results depend materially on that aggregation choice.

---

## Settings

Only the pooling strategy is changed.

Evaluated settings:

- **mean_answer**
- **last_answer**
- **mean_all**

The released runs cover the available Phase 2 medical QA configurations in this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operational change:

- `--hidden_pooling <strategy>`

Runner:

- `phase_2_medical/scripts/run_ablation_hidden_pooling.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including hidden-layer selection (`16`), bootstrap budget (`B=5000`), and cross-validation split count (`n_splits=5`). Each setting directory contains the corresponding results and manifests.

---

## Why It Matters

If the hidden probe is robust to the pooling rule, performance should remain broadly similar across the tested strategies. If meaningful differences emerge, the downstream signal depends not only on the hidden states themselves but also on how they are summarized.

This is therefore a feature-construction sensitivity check rather than an attempt to optimize pooling in isolation.

---

## Reading the Results

The main question is whether the hidden probe remains reliable under reasonable aggregation choices. The results should not be read as establishing a universally optimal pooling strategy beyond the tested settings.
