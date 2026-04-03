# Phase 1 Data Documentation (`phase_1_replication/DATA.md`)

This directory documents the Phase-1 data artifacts used in the master's thesis
"White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models".

Phase 1 investigates hallucination detection using white-box signals derived from a model's internal computations.
Uncertainty quantification in the strict probabilistic sense and calibration are addressed in later phases of the thesis.

Importantly, this repository implements a **conceptual replication** of representative white-box hallucination scoring methods proposed in prior work. Relative to the original studies, the implementation scope is partial: it reproduces core signal definitions and extraction procedures under a unified experimental setup, rather than full end-to-end systems, paper-specific training pipelines, or task-optimized detectors.

---

## Research Objective

The objective of Phase 1 is to study whether internal model signals can reliably discriminate hallucinated from non-hallucinated responses under controlled conditions. Rather than proposing a novel detection method, Phase 1 systematically compares representative white-box scoring approaches, namely logit-based confidence measures, hidden-state-based probes, and gradient-based discrepancy signals, within a unified and reproducible evaluation framework.

The task is formulated as binary classification:

- hallucinated = 1: the model answer contains at least one specific, falsifiable factually incorrect or fabricated claim
- hallucinated = 0: the model answer does not contain such a claim and is factually correct or epistemically appropriate

A single falsifiable incorrect claim suffices for a hallucination label (response-level annotation).

All methods are evaluated using AUROC as the primary performance metric.

---

## Phase-1 Benchmark Artifact

All Phase-1 experiments rely on a single frozen benchmark artifact:

`benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

This file serves as the **canonical Phase-1 evaluation dataset** and the fixed benchmark input for the reported Phase-1 experiments.

Each JSONL row contains:

- `qid`: stable question identifier
- `question`: original TruthfulQA question
- `reference_answer`: reference answer from the benchmark source
- `model_answer`: frozen model-generated answer
- `hallucinated`: manually assigned binary hallucination label (`0` or `1`)

All evaluation scripts operate directly on this file.
No additional dataset generation, text generation, or relabeling is performed during the actual Phase-1 evaluation runs.

---

## Dataset Construction Pipeline

The Phase-1 dataset was constructed through a deterministic multi-step pipeline.

### 1. Simplified TruthfulQA export

The optional provenance script

`src/prepare_truthfulqa_simplified.py`

converts a TruthfulQA source export into the local file

`benchmarks/truthfulqa_simplified.json`

This simplified file stores the benchmark content in a project-specific JSON format and serves as the source for subset construction.

### 2. Deterministic subset selection

The script

`src/prepare_truthfulqa_hallu_subset.py`

draws the Phase-1 subset from `truthfulqa_simplified.json`.

The subset construction uses:

- exact-N random sampling without replacement
- subset size `N = 300`
- fixed random seed `42`

The resulting subset membership is stored in:

`benchmarks/truthfulqa_hallu_subset_300_qids.json`

This file freezes the exact identities and order of the 300 TruthfulQA examples used in Phase 1.

### 3. Annotation-base construction

The script

`src/build_truthfulqa_hallu_base.py`

loads the fixed 300-example subset and generates deterministic model answers using the Phase-1 checkpoint:

`mistralai/Mistral-7B-Instruct-v0.2`

It writes the annotation-base artifact directly to:

`benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

At this construction stage, the field `hallucinated` is initialized as `null`.

### 4. Manual hallucination annotation

Manual hallucination labeling is performed **directly in**:

`benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

That is, the same JSONL file that contains the frozen question-answer pairs is subsequently annotated by replacing `hallucinated: null` with final binary labels (`0` or `1`).

Accordingly, the final Phase-1 benchmark artifact is both:

- the frozen model-output dataset, and
- the final manually annotated hallucination benchmark

No separate finalized label file is required in the current workflow.

---

## Labeling Protocol

Hallucination labels were assigned manually based on a predefined annotation guideline. The corresponding guideline is stored in:

`benchmarks/labeling_guidelines_hallucination.md`

The labeling rule is response-level and binary:

- a response is labeled `hallucinated = 1` if it contains at least one specific, falsifiable incorrect or unsupported claim
- otherwise it is labeled `hallucinated = 0`

This definition intentionally focuses on factual hallucination and does not attempt to capture broader response qualities such as style, usefulness, or rhetorical appropriateness.

---

## Class Distribution and Sampling Interpretation

The final archived benchmark artifact contains 300 examples with an approximately balanced class distribution:

- 149 hallucinated
- 151 non-hallucinated

Importantly, this near-balanced distribution should be interpreted as an **observed property of the final labeled artifact**, not as the result of an explicit class-balancing rule.

Subset membership was determined **before** manual hallucination labeling, using fixed-seed random subset sampling.
Phase 1 therefore does **not** rely on a documented ex-ante 50/50 balancing procedure.

---

## Reproducibility and Auditability

Phase 1 is designed to be reproducible and auditable at the artifact level.

The key reproducibility components are:

- `benchmarks/truthfulqa_simplified.json`
- `benchmarks/truthfulqa_hallu_subset_300_qids.json`
- `benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

Together, these files document:

- the benchmark source representation
- the exact sampled subset membership
- the final frozen and manually annotated evaluation artifact

Phase-1 evaluation can be reconstructed from the final JSONL benchmark artifact and the provided evaluation scripts, subject to the environment-dependent variation noted elsewhere in the repository documentation.

---

## Scope and Limitations

Phase 1 evaluates hallucination detection under a deliberately restricted experimental setting. In particular, the conceptual replication omits paper-specific downstream classifier architectures, dataset-specific tuning strategies, and generation-time intervention mechanisms. This design choice prioritizes methodological isolation and comparability of white-box signals over task-optimized performance.

Evaluation is conducted on a frozen dataset with a single model checkpoint under teacher-forced conditions. Results therefore characterize signal behavior within this controlled configuration rather than across heterogeneous models, prompts, or decoding strategies.

In addition, exact reconstruction of the historical model answers may depend on the original software and model environment. The archived file `truthfulqa_hallu_frozen_model_outputs_300.jsonl` should therefore be treated as the canonical benchmark artifact for evaluation and thesis reporting.

Results are intended for methodological comparison of white-box signals and do not constitute leaderboard claims.

---

## License and Usage

This repository is intended for academic and research use.
Users are responsible for complying with the licenses of external datasets and model checkpoints.
