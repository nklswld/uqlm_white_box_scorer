# Hidden Pooling Ablation

This directory contains an appendix ablation for pooling strategies
used in hidden-state feature extraction.

---

## Hypothesis

Hidden-state pooling strategy does not materially alter
hallucination detection performance once layer selection is fixed.

---

## Ablation Design

Two pooling strategies are evaluated:

- **mean_answer**
- **last_answer**

---

## Implementation

The ablation is implemented by varying the pooling strategy
used in hidden-state aggregation.

Relevant flags:

- `--hidden_pooling <strategy>`
- `--ablation_name hidden_pooling`
- `--ablation_setting <strategy>`

---

## Outputs

Results are reported as supplementary robustness checks.