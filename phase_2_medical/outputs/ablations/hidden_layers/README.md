# Hidden Layer Selection Ablation

This directory contains the ablation study for hidden-layer selection
in the hidden-state probe evaluated in Phase 2.

The ablation investigates how the choice of representation depth
affects hallucination detection performance.

---

## Hypothesis

Hidden representations from later or aggregated layers
encode more task-relevant information for error detection
than early-layer representations.

Multi-layer aggregation is expected to improve robustness
by reducing layer-specific variance.

---

## Ablation Design

The following layer configurations are evaluated:

- **L_early**: early transformer layer  
- **L_mid**: middle transformer layer  
- **L_late**: late transformer layer  
- **L_multi_-4_-3_-2_-1**: aggregation of the final four layers  

Layer indices are defined relative to the model architecture
and are kept consistent across all runs.

---

## Implementation

The ablation is implemented by varying the hidden-layer indices
used for feature extraction in the hidden-state probe.

Relevant flags:

- `--hidden_layers <indices>`
- `--ablation_name hidden_layers`
- `--ablation_setting <layer_setting>`

All probes are trained and evaluated using leakage-safe
out-of-fold predictions.

---

## Outputs

Each setting directory contains the full set of evaluation artifacts,
including bootstrap confidence intervals and coverage statistics.