# White-Box Hallucination Detection and Uncertainty Scoring in LLMs

This repository contains the full implementation used in a master’s thesis:

**“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models.”**

The project is structured into two experimental phases:

- **Phase 1 (`phase_1_replication/`)**: controlled replication of white-box hallucination signals on TruthfulQA
- **Phase 2 (`phase_2_medical/`)**: transfer evaluation of selected white-box signals to constrained medical QA (MedQA, PubMedQA)

---

## Project Scope

The codebase focuses on **post-hoc white-box scoring** and **discriminative evaluation**.

In scope:

- logit-based token confidence signals (e.g., LNTP, MTP)
- gradient-based internal discrepancy signals
- hidden-state probe-based scoring
- leakage-safe out-of-fold evaluation for supervised probes
- reproducible metric reporting with bootstrap confidence intervals

Out of scope:

- model fine-tuning
- calibration pipelines for probabilistic uncertainty estimates
- generation-time mitigation/intervention methods

---

## Repository Structure

- `phase_1_replication/`  
  Phase-1 implementation, frozen TruthfulQA artifacts, reproducibility docs

- `phase_2_medical/`  
  Phase-2 medical QA implementation, frozen prediction artifacts, analysis scripts

- `requirements.txt`  
  Shared Python dependencies

- `benchmarks_overview.txt`  
  High-level benchmark reference notes

---

## Phase Summaries

## Phase 1: Replication on TruthfulQA

Goal:

- evaluate whether internal white-box signals separate hallucinated vs non-hallucinated responses

Primary docs:

- `phase_1_replication/README.md`
- `phase_1_replication/DATA.md`
- `phase_1_replication/reproduce_phase1.md`

Main entrypoint:

```bash
python phase_1_replication/src/run_phase1_truthfulqa.py
```

---

## Phase 2: Medical QA Transfer

Goal:

- test whether Phase-1 white-box signals generalize to constrained medical QA error detection

Primary docs:

- `phase_2_medical/README.md`
- `phase_2_medical/DATA.md`
- `phase_2_medical/reproduce_phase2.md`

Main baseline runner:

```bash
bash phase_2_medical/scripts/run_baseline_all.sh
```

Analysis:

```bash
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

---

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Reproduce one phase:

- Phase 1: follow `phase_1_replication/reproduce_phase1.md`
- Phase 2: follow `phase_2_medical/reproduce_phase2.md`

3. Use archived outputs where available for exact thesis-level reproducibility.

---

## Reproducibility Principles

Across both phases, the project emphasizes:

- fixed seeds for controlled randomness
- frozen benchmark/prediction artifacts for stable post-hoc evaluation
- deterministic analysis scripts
- serialized bootstrap index artifacts for CI regeneration
- manifest files that capture run configuration and software metadata

Small floating-point differences across hardware/runtime stacks can still occur.

---

## Outputs Overview

Typical output categories in phase subdirectories:

- `outputs/final/`  
  canonical run artifacts (`results`, `manifest`, `bootstrap_indices`)

- `outputs/figures_tables/`  
  exported tables and figures

- `outputs/ablations/`  
  optional ablation experiments (primarily in Phase 2)

---

## Recommended Reading Order

1. `phase_1_replication/README.md`
2. `phase_1_replication/reproduce_phase1.md`
3. `phase_2_medical/README.md`
4. `phase_2_medical/reproduce_phase2.md`
5. `phase_2_medical/DATA.md`

---

## License and Usage

This repository is intended for academic and research use.

Users are responsible for complying with licenses and usage terms of external datasets, model checkpoints, and any third-party resources.