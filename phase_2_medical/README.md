# White-Box Hallucination Detection for Large Language Models (Phase 2)

This directory contains the complete implementation of **Phase 2** of the master’s thesis  
**“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models.”**

Phase 2 evaluates whether white-box hallucination/uncertainty signals validated in Phase 1
generalize to constrained medical QA tasks.

---

## Research Objective

The objective of Phase 2 is to assess robustness and transferability of white-box signals across:

- task formats (multiple-choice vs constrained short-answer),
- datasets (MedQA vs PubMedQA),
- and model families (Mistral vs BioMistral).

Hallucination detection is operationalized as binary error detection:

- `error = 1`: model-selected answer is incorrect
- `error = 0`: model-selected answer is correct

Primary metric: **AUROC**  
Secondary metric: **Spearman’s rho**

---

## Methods and Signals

Phase 2 evaluates the following scorers:

- **LNTP** (Length-Normalized Token Probability)
- **MTP** (Minimum Token Probability)
- **EGH probe (OOF)**
- **Hidden-state probe (OOF)**

Notes:

- Logit-based scores are treated as ranking signals (not calibrated probabilities).
- Supervised probes are evaluated via out-of-fold predictions to avoid leakage.
- Score orientation is checked before metric reporting for consistent interpretation.

---

## Canonical Artifacts (Source of Truth)

The reported Phase-2 results are anchored to archived artifacts in `outputs/final/`:

- `*.results.jsonl`
- `*.manifest.json`
- `*.manifest.bootstrap_indices.npz`

These are the canonical evaluation artifacts used for all reported tables/figures.

---

## Datasets and Frozen Evaluation Setup

Phase 2 uses frozen subsets of:

- **MedQA** (USMLE-style multiple-choice)
- **PubMedQA** (biomedical yes/no/maybe)

For each dataset:

- fixed subset size: `N = 1000`
- frozen labels and frozen model outputs
- no dataset-specific tuning during evaluation

Benchmark files and schema details are documented in [`DATA.md`](./DATA.md).

---

## Models

Phase 2 evaluates two 7B instruction-tuned models:

- `mistralai/Mistral-7B-Instruct-v0.2`
- `BioMistral/BioMistral-7B`

Model outputs are generated once and reused unchanged for scoring and analysis.

---

## Evaluation Protocol

Phase 2 follows a controlled post-hoc protocol:

- fixed global seed (`42`) for default runs
- frozen-output regime
- teacher-forced white-box extraction on frozen prompt-answer pairs
- stratified bootstrap confidence intervals (`B = 5000`)
- serialized bootstrap indices for auditability and CI regeneration

Token-budget constraints are fixed in baseline runs (task-specific settings; see
[`reproduce_phase2.md`](./reproduce_phase2.md)).

---

## Reproduction

For complete, step-by-step reproduction (exact commands, parameters, paths), see:

- [`reproduce_phase2.md`](./reproduce_phase2.md)

Quick entry points:

Run baseline experiments:
```bash
bash phase_2_medical/scripts/run_baseline_all.sh
```

Generate tables and figures:
```bash
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

---

## Analysis and Visualization

Main analysis scripts:

- `analysis/phase2_tables.py`
- `analysis/phase2_figures.py`

Outputs are written to:

- `outputs/figures_tables/tables_general/`
- `outputs/figures_tables/figures_general/`

Ablation outputs are stored under `outputs/ablations/`.

---

## Repository Structure (Phase 2)

- `benchmarks/`  
  Frozen benchmark subsets and labels

- `src/`  
  Data preparation, frozen-generation, and scoring pipeline

- `scripts/`  
  Reproducible shell entrypoints (baseline and ablations)

- `analysis/`  
  Deterministic table/figure generation and metric aggregation

- `outputs/`  
  Final artifacts, manifests, bootstrap indices, ablation outputs, figures/tables

- `reproduce_phase2.md`  
  Full reproduction instructions

- `DATA.md`  
  Dataset/schema documentation

---

## Scope and Limitations

Phase 2 evaluates hallucination detection as incorrect answer selection in constrained medical QA.
It does **not** model free-form hallucination generation.

Results should be interpreted as evidence of discriminative separability under the frozen-output,
task-constrained regime (not as calibrated uncertainty estimation or intervention performance).

---

## License and Usage

This repository is intended for academic and research use.

Users are responsible for compliance with licenses of external datasets and model checkpoints.