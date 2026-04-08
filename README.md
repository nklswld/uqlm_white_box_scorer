# White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models

This repository accompanies a master's thesis on white-box scoring methods for large language models. It combines research code, archived outputs, benchmark subsets, and reproduction notes for two linked experimental phases.

The repository is organized as an academic artifact/codebase, not as a production package. In most cases, the most useful entry point is the archived outputs and the phase-specific documentation rather than a full rerun.

## Overview

- `phase_1_replication/`: conceptual replication of white-box hallucination scoring methods on frozen TruthfulQA answers
- `phase_2_medical/`: transfer of selected white-box signals to constrained medical QA, framed as benchmark-defined prediction error detection on MedQA and PubMedQA

Across the two phases, the implemented methods include intrinsic logit-based scores such as LNTP and MTP, plus supervised white-box readouts based on hidden states and EGH-style feature families.

## Where to Start

If you want to understand the repository quickly, start with:

1. `phase_1_replication/README.md`
2. `phase_2_medical/README.md`
3. `benchmarks_overview.md`
4. `ARTIFACT_MANIFEST.md`

For reproduction and data details, see:

- `phase_1_replication/reproduce_phase1.md`
- `phase_1_replication/DATA.md`
- `phase_2_medical/reproduce_phase2.md`
- `phase_2_medical/DATA.md`

## Repository Layout

```text
phase_1_replication/
  benchmarks/   frozen TruthfulQA subset, labels, and annotation material
  src/          Phase 1 scoring pipeline
  analysis/     figure generation
  outputs/      archived results, manifests, and PDFs

phase_2_medical/
  benchmarks/   prepared MedQA and PubMedQA subsets
  src/          frozen-generation and scoring pipeline
  scripts/      maintained baseline and ablation runners
  analysis/     table and figure generation
  outputs/      frozen predictions, final baseline runs, ablations, figures, tables
```

Useful root-level files:

- `benchmarks_overview.md`: benchmark summary across both phases
- `ARTIFACT_MANIFEST.md`: inventory of the main archived artifacts
- `requirements.txt`: shared Python dependencies
- `CITATION.cff`: citation metadata

## Main Artifacts

Some of the most useful files and directories are:

- `phase_1_replication/benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`
- `phase_1_replication/outputs/phase1_truthfulqa_hallu_results_300.jsonl`
- `phase_1_replication/outputs/phase1_run_manifest.json`
- `phase_1_replication/outputs/self_iaa_summary.json`
- `phase_1_replication/outputs/figs/`
- `phase_2_medical/outputs/frozen/`
- `phase_2_medical/outputs/final/`
- `phase_2_medical/outputs/figures_tables/`
- `phase_2_medical/outputs/ablations/`

If you mainly want the released Phase 2 baseline results, start with `phase_2_medical/outputs/final/`. If you want the derived figures and tables, go to `phase_2_medical/outputs/figures_tables/`.

## Common Tasks

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Regenerate analysis outputs from archived results:

```bash
python phase_1_replication/analysis/phase1_figures.py
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

Rerun the main pipelines:

```bash
python phase_1_replication/src/run_phase1_truthfulqa.py
bash phase_2_medical/scripts/run_baseline_all.sh
```

Phase 2 also includes maintained ablation runners under `phase_2_medical/scripts/`.

## Environment Notes

- Python `3.10+`
- GPU is recommended for the heavier extraction and scoring steps
- full reruns may require access to models or datasets hosted via Hugging Face
- the maintained Phase 2 shell runners assume a Linux-like environment such as Linux or WSL

## Citation and License

Citation metadata is provided in `CITATION.cff`.

The repository is licensed under `CC BY-NC 4.0`; see `LICENSE`. External datasets and model checkpoints referenced by this project remain subject to their own licenses and terms.