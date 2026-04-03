# White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models (Phase 2)

This directory documents **Phase 2** of the master's thesis **"White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models."**

Phase 2 evaluates whether white-box signals that are diagnostically useful in the Phase-1 hallucination setting retain discriminative value in constrained medical QA under benchmark-defined correctness targets.

---

## Research Objective

Phase 2 examines how selected white-box signals behave across:

- task formats (multiple-choice vs constrained short-answer),
- datasets (MedQA vs PubMedQA),
- and model families (Mistral vs BioMistral).

The primary Phase-2 target is **benchmark-defined prediction error detection**:

- `error = 1`: model-selected answer is incorrect
- `error = 0`: model-selected answer is correct

This preserves the binary detection structure of Phase 1, but not its exact semantic target. Phase 2 does not provide a direct measure of response-level factual hallucination in unrestricted medical generation.

Primary metric: **AUROC**  
Secondary metric: **Spearman's rho**

---

## Methods and Signals

Phase 2 evaluates the following scorers:

- **LNTP** (Length-Normalized Token Probability; intrinsic score)
- **MTP** (Minimum Token Probability; intrinsic score)
- **EGH probe (OOF)** (supervised readout on white-box features)
- **Hidden-state probe (OOF)** (supervised readout on pooled hidden representations)

Interpretation notes:

- intrinsic logit-based scores are used as ranking signals rather than calibrated probabilities
- supervised probes are evaluated via out-of-fold predictions to avoid leakage
- score orientation is checked before metric reporting

---

## Canonical Artifacts (Source of Truth)

The reported Phase-2 results are anchored to archived artifacts in `outputs/final/`:

- `*.results.jsonl`
- `*.manifest.json`
- `*.manifest.bootstrap_indices.npz`

These files anchor the reported Phase-2 tables and figures.

---

## Datasets and Frozen Evaluation Setup

Phase 2 uses fixed 1,000-example subsets of:

- **MedQA** (USMLE-style multiple-choice)
- **PubMedQA** (biomedical yes/no/maybe)

For each dataset, the evaluation uses frozen labels and frozen model outputs, with no dataset-specific tuning during scorer evaluation.

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

Operationally:

- frozen model continuations are generated once and then rescored post hoc under teacher forcing
- constrained parsing maps free-form continuations into the benchmark label space before error evaluation
- the released Markdown documentation records the protocol structure, but not the full underlying prompt text

---

## Reproduction

For the documented reproduction procedure, command paths, and expected outputs, see:

- [`reproduce_phase2.md`](./reproduce_phase2.md)

The maintained batch runners assume a Linux-like shell environment.

Common entry points:

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

## Analysis Outputs

Main analysis scripts:

- `analysis/phase2_tables.py`
- `analysis/phase2_figures.py`

Baseline summary exports are written to:

- `outputs/figures_tables/tables_general/`
- `outputs/figures_tables/figures_general/`

Ablation run artifacts are stored under `outputs/ablations/`. Derived ablation summaries are written under `outputs/figures_tables/ablations/`.

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

Phase 2 evaluates benchmark-defined incorrect answer selection in constrained medical QA.
It does **not** directly evaluate free-form medical hallucination generation in the Phase-1 sense.

Results should be read as evidence about ranking-based discrimination in a frozen, task-constrained setting, not as calibrated uncertainty estimation or intervention performance.

---

## License and Usage

This repository is intended for academic and research use.

Users are responsible for compliance with licenses of external datasets and model checkpoints.
