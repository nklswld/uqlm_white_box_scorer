# Hidden Pooling Ablation

## Overview

This directory contains a methodological ablation examining the effect of hidden-state pooling strategy on the hidden-state probe used in the Phase 2 medical QA evaluation.

The ablation studies how token-level hidden representations are aggregated into a single example-level feature representation for the hidden probe. Its purpose is to determine whether probe performance is materially affected by this aggregation choice.

---

## Hypothesis

Once the hidden-layer selection is fixed, hidden-probe performance is expected to be relatively stable across reasonable pooling strategies. Any observed differences should be secondary to the underlying representational content rather than the aggregation rule itself.

---

## Motivation / Background

Phase 2 evaluates white-box scores for uncertainty quantification and hallucination or error detection in constrained medical QA tasks. For the hidden-state probe, token-level representations must be pooled into a fixed-size example representation before supervised evaluation can be performed.

Pooling strategy may influence which aspects of the hidden-state signal are preserved or emphasized. This ablation therefore tests whether the hidden probe is robust to this representation-construction choice, or whether its effectiveness depends on a specific pooling convention.

---

## Ablation Design

The ablation varies only the hidden-state pooling strategy used in the hidden probe.

Evaluated settings:

- **mean_answer**
- **last_answer**
- **mean_all**

The runs in this directory cover the Phase 2 medical QA configurations provided for this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operationally, the ablation is implemented by changing:

- `--hidden_pooling <strategy>`

The maintained batch runner for this experiment is:

- `phase_2_medical/scripts/run_ablation_hidden_pooling.sh`

In the maintained sweep, the remaining hidden-probe and evaluation settings are held fixed, including hidden-layer selection (`16`), bootstrap budget (`B=5000`), and cross-validation split count (`n_splits=5`). Each setting directory contains the corresponding run artifacts, including result files and manifests for direct comparison across pooling strategies.

---

## Expected Effect

If the hidden probe is robust to the pooling rule, performance should remain broadly similar across the tested strategies, with only limited variation. If meaningful differences emerge, this would indicate that the way hidden states are aggregated materially affects the downstream detection signal.

For uncertainty quantification and hallucination detection, this matters because sensitivity to pooling strategy would imply that the hidden-state score depends not only on the presence of useful internal information, but also on how that information is summarized into probe features.

---

## Notes / Interpretation

This ablation should be interpreted as a sensitivity analysis of feature construction within the hidden-state probe. It is not intended to establish a universally optimal pooling strategy beyond the tested settings.

Accordingly, the main question is whether the hidden probe remains reliable under reasonable aggregation choices, or whether conclusions depend strongly on a particular pooling convention.