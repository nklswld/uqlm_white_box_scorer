# OOF Seed Robustness Ablation

This directory contains the ablation study for robustness
of supervised probes with respect to random seed variation.

---

## Hypothesis

Out-of-fold probe performance is stable with respect to
reasonable variations in the random seed used for
cross-validation splits.

---

## Ablation Design

Multiple global seeds are evaluated:

- **seed0**
- **seed42**
- **seed123**

All other hyperparameters and data splits are held constant.

---

## Implementation

The ablation is implemented by varying the global random seed
used for cross-validation and bootstrap resampling.

Relevant flags:

- `--seed <value>`
- `--ablation_name robustness_oof`
- `--ablation_setting seed<value>`

---

## Outputs

Each setting directory contains identical evaluation artifacts,
allowing direct comparison of stability across seeds.