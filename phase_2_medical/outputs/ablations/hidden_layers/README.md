# Hidden Layer Selection Ablation

## Aim

This ablation varies the transformer layer used to extract hidden-state features for the supervised hidden probe in Phase 2. It addresses representation depth directly: the question is how much the hidden-probe results depend on which internal layer is observed.

---

## Settings

Only the hidden-layer index is changed.

Evaluated settings:

- **layer_4**
- **layer_16**
- **layer_24**
- **layer_32**

The released runs cover the available Phase 2 medical QA configurations in this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operational change:

- `--hidden_layers <indices>`

Runner:

- `phase_2_medical/scripts/run_ablation_hidden_layers.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including pooling strategy (`mean_answer`), bootstrap budget (`B=5000`), and cross-validation split count (`n_splits=5`). Each setting directory contains the corresponding results and manifests.

---

## Why It Matters

Hidden representations change substantially across network depth. If the hidden probe is robust, results should remain broadly stable across the tested layers. If layer choice matters strongly, clearer differences should emerge across the evaluated depths.

Strong sensitivity would indicate that the hidden-state signal depends materially on a representation choice rather than reflecting a uniformly available internal error signal.

---

## Reading the Results

This ablation is a sensitivity check on representation choice within the hidden-state probe. It is not evidence that deeper layers are universally superior, and it should not be read as an architecture-independent layer-selection rule.
