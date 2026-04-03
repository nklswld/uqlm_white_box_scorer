# Bootstrap Budget Ablation

## Aim

This ablation varies the bootstrap budget `B` used to estimate confidence intervals in the Phase 2 medical QA evaluation. It addresses interval stability rather than scorer quality: the question is whether the reported intervals change materially when the number of resamples is increased.

---

## Settings

Only the bootstrap budget is changed.

Evaluated settings:

- **B1000**
- **B2000**
- **B5000**
- **B10000**

The released runs cover the available Phase 2 medical QA configurations in this ablation, including MedQA and PubMedQA with Mistral and BioMistral variants.

Operational change:

- `--B <value>`

Runner:

- `phase_2_medical/scripts/run_ablation_bootstrap_budget.sh`

Each setting directory contains the corresponding results, manifests, and persisted bootstrap index files.

---

## Why It Matters

If the baseline choice of `B` is already adequate, interval endpoints and widths should change only modestly as the budget increases. If they move substantially, the reported uncertainty bands are still sensitive to Monte Carlo approximation error.

This is therefore a check on statistical estimation stability, not an attempt to improve underlying scorer performance.

---

## Reading the Results

The main comparison is the stability of confidence intervals across `B`, especially their width and endpoints. Changes in scorer ranking are secondary. If moderate budgets already yield stable intervals, larger budgets mainly add compute cost. If substantial variation remains, the corresponding interval estimates should be treated more cautiously.
