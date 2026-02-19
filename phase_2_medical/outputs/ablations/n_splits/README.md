# Cross-Validation Splits Ablation

This directory contains an appendix ablation studying
the sensitivity of probe performance to the number of
cross-validation folds.

---

## Hypothesis

Probe performance is stable across reasonable choices
of cross-validation fold count.

---

## Ablation Design

The following values are evaluated:

- **K3**
- **K5**
- **K10**

---

## Implementation

The ablation is implemented by varying the number of
cross-validation splits.

Relevant flags:

- `--n_splits <value>`
- `--ablation_name n_splits`
- `--ablation_setting K<value>`

---

## Outputs

Results are included in the appendix as a stability check.