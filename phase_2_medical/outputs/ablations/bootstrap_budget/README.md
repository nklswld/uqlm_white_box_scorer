# Bootstrap Budget Ablation

## Overview

This directory contains a methodological ablation examining the effect of the bootstrap budget `B` on the confidence intervals reported in the Phase 2 medical QA evaluation.

In this context, `B` denotes the number of bootstrap resamples used to estimate uncertainty around scorer-level evaluation metrics. The purpose of the ablation is to assess whether the reported interval estimates are numerically stable across reasonable choices of bootstrap budget.

---

## Hypothesis

Bootstrap-based confidence intervals should become more stable as the number of resamples increases. Beyond a moderate bootstrap budget, further increases in `B` are expected to yield diminishing returns in interval precision.

---

## Motivation / Background

Phase 2 evaluates white-box scores for uncertainty quantification and hallucination or error detection in constrained medical QA settings. Because the reported evaluation includes bootstrap confidence intervals, the numerical reliability of these intervals depends in part on the resampling budget.

This ablation therefore serves as a robustness check on the uncertainty-estimation procedure rather than on the underlying scoring methods themselves. It asks whether the selected bootstrap budget is sufficient to support stable and interpretable interval estimates without unnecessary computational cost.

---

## Ablation Design

The ablation varies only the bootstrap budget `B`, that is, the number of bootstrap resamples used for interval estimation.

Evaluated settings:

- **B1000**
- **B2000**
- **B5000**
- **B10000**

The runs in this directory cover the Phase 2 medical QA configurations provided for this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operationally, the ablation is implemented by changing:

- `--B <value>`

The maintained batch runner for this experiment is:

- `phase_2_medical/scripts/run_ablation_bootstrap_budget.sh`

Each setting directory contains the corresponding run artifacts, including result files, manifests, and persisted bootstrap index files for direct comparison across budgets.

---

## Expected Effect

If the bootstrap budget is sufficiently large, the estimated confidence intervals should show only minor changes when `B` is increased further. In particular, interval widths and interval endpoints should converge as Monte Carlo noise from the bootstrap approximation decreases.

For uncertainty quantification and hallucination detection, this matters because unstable confidence intervals can make reported uncertainty estimates appear more precise or less precise than warranted. A stable bootstrap budget supports more reliable interpretation of scorer comparisons and robustness claims.

---

## Notes / Interpretation

This ablation should be interpreted primarily as a sensitivity analysis of statistical estimation. It is not intended to demonstrate improved scorer quality or improved task performance through larger bootstrap budgets.

Accordingly, the main quantities of interest are the stability and width of the reported confidence intervals, rather than changes in the underlying scorer rankings or point estimates. If results are already stable at moderate `B`, then larger budgets mainly increase computational cost. If substantial variation remains across budgets, the corresponding uncertainty estimates should be interpreted more cautiously.