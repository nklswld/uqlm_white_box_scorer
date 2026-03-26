# Hidden Layer Selection Ablation

## Overview

This directory contains a methodological ablation examining the effect of hidden-layer selection on the hidden-state probe used in the Phase 2 medical QA evaluation.

The ablation focuses on representation depth, that is, which transformer layer is used to extract hidden-state features for the supervised hidden probe. Its purpose is to assess how sensitive hidden-probe performance is to the choice of internal representation.

---

## Hypothesis

Hidden-probe performance is expected to depend on representation depth. Relative to early layers, middle or later layers may provide more task-relevant information for distinguishing correct from erroneous model outputs.

---

## Motivation / Background

Phase 2 evaluates white-box scores for uncertainty quantification and hallucination or error detection in constrained medical QA tasks. Among these scores, the hidden-state probe is unique in that it depends directly on internal model representations rather than only on output-space quantities.

Because hidden representations vary across network depth, the choice of extraction layer is a consequential design decision. This ablation therefore tests whether the hidden probe is robust to reasonable layer choices, or whether its effectiveness depends strongly on a specific depth.

---

## Ablation Design

The ablation varies only the hidden-layer index used for hidden-state feature extraction in the hidden probe.

Evaluated settings:

- **layer_4**
- **layer_16**
- **layer_24**
- **layer_32**

The runs in this directory cover the Phase 2 medical QA configurations provided for this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operationally, the ablation is implemented by changing:

- `--hidden_layers <indices>`

The maintained batch runner for this experiment is:

- `phase_2_medical/scripts/run_ablation_hidden_layers.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including pooling strategy (`mean_answer`), bootstrap budget (`B=5000`), and cross-validation split count (`n_splits=5`). Each setting directory contains the corresponding run artifacts, including result files and manifests for direct comparison across layers.

---

## Expected Effect

If the hidden probe is robust to representation depth, performance should remain broadly stable across the tested layer settings, with at most moderate variation. If layer choice is important, clearer differences should emerge across the evaluated depths.

For uncertainty quantification and hallucination detection, this matters because strong sensitivity to layer selection would indicate that the hidden-state signal depends materially on a representational design choice rather than reflecting a uniformly available internal error signal.

---

## Notes / Interpretation

This ablation should be interpreted as a sensitivity analysis of representation choice within the hidden-state probe. It is not intended to show that deeper layers are universally superior, nor to optimize architecture-specific layer selection beyond the tested configurations.

Accordingly, the main question is whether the hidden probe remains effective across reasonable layer choices, or whether conclusions depend strongly on a narrow representation-depth setting.