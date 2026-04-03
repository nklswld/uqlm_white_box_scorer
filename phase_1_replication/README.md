# White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models (Phase 1)

This directory documents Phase 1 of the master's thesis **"White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models."**

Phase 1 investigates hallucination detection using white-box signals derived from a model's internal computations.
Uncertainty quantification in the strict probabilistic sense and calibration are addressed in later phases of the thesis.

---

## Research Objective

Phase 1 examines whether internal model signals provide useful ranking information for hallucination detection beyond surface-level confidence proxies.

The task is formulated as binary classification:

- hallucinated = 1: the model answer contains at least one factually incorrect or fabricated claim
- hallucinated = 0: the model answer is factually correct or epistemically appropriate

Primary evaluation metric: **AUROC**

---

## Methods and Signals

Phase 1 evaluates three classes of white-box methods on the TruthfulQA benchmark:

- logit-based methods
- gradient-based methods
- hidden-state-based methods

Implemented scorers include:

- LNTP (Length-Normalized Token Probability)
- MTP (Minimum Token Probability)
- gradient-based EGH-inspired primitives (e.g., KL gap, gradient norm, embedding differences)
- hidden-state-based logistic regression probe

EGH primitives are evaluated both as standalone ranking scores and, where configured, as feature inputs to a supervised out-of-fold probe.

Logit-based scores represent token-level confidence signals derived from model likelihoods and are used as ranking scores for hallucination detection.
They are treated as inverse uncertainty proxies but are not probabilistic uncertainty estimates.

Supervised scorers are evaluated using leakage-safe out-of-fold predictions.

---

## Replication Scope

Phase 1 should be understood primarily as a **conceptual replication** of representative white-box hallucination scoring approaches under a unified evaluation framework.

Relative to the original studies, the implemented scope is partial: Phase 1 reproduces core signal extraction and discriminative evaluation logic, but not the full paper-specific training pipelines or task-optimized end-to-end systems.

The Phase-1 comparison draws on components from prior work:

- Liu et al. (2024): hidden-state representations for correctness prediction
- Bouchard et al. (2025): logit-based token probability scores
- Hu et al. (2024): gradient-based white-box signals

The comparison tests whether the central qualitative signal patterns reported in prior work persist under a controlled, common hallucination-detection protocol.

Not in scope for Phase 1:

- calibration methods and uncertainty estimation beyond hallucination detection
- cross-dataset generalization or domain transfer
- end-to-end hallucination mitigation

---

## Dataset and Frozen Evaluation Setup

Evaluation is performed on a frozen subset of the TruthfulQA benchmark.

The final Phase-1 benchmark artifact is:

`benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

This file serves as the single source of truth for Phase 1 and contains:

- `qid`
- `question`
- `reference_answer`
- `model_answer`
- `hallucinated`

Model outputs are generated once using a fixed prompt and checkpoint and are then reused unchanged across all experiments.
All Phase-1 evaluations use teacher forcing only; no text generation is performed during evaluation.

Model checkpoint used in Phase 1:

`mistralai/Mistral-7B-Instruct-v0.2`

The underlying 300-example subset is sampled reproducibly with fixed seed `42` and stored in:

`benchmarks/truthfulqa_hallu_subset_300_qids.json`

Further dataset and construction details are documented in `DATA.md`.

---

## Data Construction Workflow

The Phase-1 benchmark can be reconstructed through the following preparation steps:

1. `src/prepare_truthfulqa_simplified.py`
   converts a TruthfulQA source export into
   `benchmarks/truthfulqa_simplified.json`

2. `src/prepare_truthfulqa_hallu_subset.py`
   draws the deterministic 300-example subset and writes
   `benchmarks/truthfulqa_hallu_subset_300_qids.json`

3. `src/build_truthfulqa_hallu_base.py`
   generates frozen model answers for the fixed subset and writes
   `benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`
   with `hallucinated = null`

4. manual annotation is then performed directly in
   `benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`

For evaluation and thesis-level verification, only the final frozen benchmark artifact is required.

---

## Human Labeling and Reliability

Hallucination labels were assigned manually based on a predefined annotation guideline:

`benchmarks/labeling_guidelines_hallucination.md`

Labeling reliability was assessed using a self-inter-annotator agreement procedure:

- stratified subsample of 80 examples
- second blinded annotation round
- Cohen's kappa with bootstrap confidence intervals

Results:

- Agreement: 92.5%
- Cohen's kappa: 0.85
- 95% confidence interval: [0.72, 0.95]

Full results are stored in:

`outputs/self_iaa_summary.json`

---

## Evaluation Protocol

Phase 1 uses a leakage-robust evaluation protocol:

- fixed global random seed (42)
- 5-fold stratified cross-validation
- supervised models evaluated using out-of-fold predictions
- unsupervised scores computed once per sample
- AUROC as the primary metric
- stratified bootstrap confidence intervals for stability assessment

In addition to absolute AUROC values, the evaluation records AUROC differences relative to a random baseline (`constant score = 0.5`) via paired bootstrap resampling.

Evaluation artifacts, including bootstrap resampling indices where available, are written to disk for auditability.

---

## Reproducibility

Phase 1 is reproduced from frozen inputs and deterministic scripts.

Run evaluation:

```bash
pip install -r requirements.txt
python src/run_phase1_truthfulqa.py
```

Primary run outputs:

- `outputs/phase1_truthfulqa_hallu_results_300.jsonl`
- `outputs/phase1_run_manifest.json`

If bootstrap delta comparisons are enabled locally, an additional file may be written:

- `outputs/phase1_run_manifest.bootstrap_indices.npz`

Notes:

- The canonical archived Phase-1 statistics can be verified directly from `outputs/phase1_run_manifest.json`.
- Phase-1 bootstrap index files are not tracked in Git in the current repository state.
- Exact CI replay from serialized bootstrap indices therefore requires the local artifact bundle used during the original run or the accompanying thesis submission archive.

---

## Environment

Minimum environment:

- Python `>= 3.10`
- dependencies installed from repository root via `pip install -r requirements.txt`

Execution notes:

- CPU execution is possible but slower.
- GPU execution is recommended for hidden-state extraction workloads.
- Reference experiments were executed on an NVIDIA A100 (80GB).

---

## Determinism and Auditability

Phase 1 fixes the global seed, cross-validation splits, and bootstrap resampling procedure.

Repeated reruns are therefore intended to reproduce the archived results up to minor floating-point variation across hardware or runtime stacks. The run manifest records configuration and aggregate metrics, and the example-level results file stores the per-sample scores used for evaluation.

---

## Scope and Limitations

Phase 1 is a controlled post-hoc evaluation on frozen model outputs.

- No text generation occurs during scorer evaluation.
- No fine-tuning or parameter updates are performed.
- Results should be interpreted as methodological comparison of intrinsic scores and supervised white-box readouts under a fixed benchmark setup, not as evidence of calibration or deployment-time utility.

Exact reconstruction of the historical frozen answers may depend on the original software and model environment. For thesis-level reporting and audit, `benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl` and the archived outputs under `outputs/` remain the reference artifacts.

---

## Related Documentation

- `DATA.md`
- `reproduce_phase1.md`
- `benchmarks/labeling_guidelines_hallucination.md`
