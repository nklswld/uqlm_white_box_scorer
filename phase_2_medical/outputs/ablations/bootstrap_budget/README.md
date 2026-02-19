# Bootstrap Budget Ablation

This directory contains an appendix ablation studying
the effect of bootstrap sample count on confidence intervals.

---

## Hypothesis

Bootstrap confidence intervals stabilize beyond a moderate
number of resamples, with diminishing returns for larger budgets.

---

## Ablation Design

The following bootstrap budgets are evaluated:

- **B1000**
- **B5000**
- **B10000**

---

## Implementation

The ablation is implemented by varying the bootstrap sample count.

Relevant flags:

- `--B <value>`
- `--ablation_name bootstrap_budget`
- `--ablation_setting B<value>`

---

## Outputs

Results validate the numerical stability of reported
confidence intervals.