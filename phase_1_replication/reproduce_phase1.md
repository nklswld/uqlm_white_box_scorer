# reproduce_phase1.md

## Purpose

This document describes how to reproduce all Phase-1 results in this repository.

Phase 1 evaluates white-box hallucination detection methods on a frozen subset of the TruthfulQA benchmark.
All experiments operate exclusively on pre-generated model outputs and are fully deterministic.

No text generation, sampling, fine-tuning, or labeling is performed during reproduction.

---

## Frozen Outputs

Phase-1 experiments rely on a single frozen dataset file containing model answers generated once and never modified.

Prompt template used during generation:

Question: {question}  
Answer:

No additional system prompts, role tokens, or trailing whitespace were used.  
Tokenization was performed with `add_special_tokens=False`.

All Phase-1 computations use teacher forcing only.

---

## Dataset

Required file:

benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl

Each line is a JSON object with the following fields:

- qid: unique question identifier  
- question: question text  
- reference_answer: ground-truth answer  
- model_answer: frozen model output  
- hallucinated: binary hallucination label (0/1)  

No other datasets are required.

---

## Environment

- Python >= 3.10  
- Dependencies installed via:  
  pip install -r requirements.txt  

Tested with GPU and CPU execution.  
GPU is recommended for faster hidden-state extraction.

---

## Reproduction

Verify repository structure:

phase_1_replication/  
- src/  
- benchmarks/  
- outputs/  

Run Phase-1 evaluation:

python src/run_phase1_truthfulqa.py

The script performs:

- loading frozen TruthfulQA samples  
- computation of unsupervised white-box scores  
- extraction of hidden-state features  
- supervised evaluation using 5-fold stratified out-of-fold (OOF) cross-validation  
- AUROC computation  
- stratified bootstrap confidence intervals  
- writing results and a run manifest  

No separate held-out test split is used.  
All supervised scorers are evaluated on the full dataset using leakage-safe OOF predictions.

---

## Outputs

After execution, the following files are created:

outputs/phase1_truthfulqa_hallu_results_300.jsonl  
outputs/phase1_run_manifest.json  

If bootstrap delta comparisons are enabled, an additional file is written:

outputs/phase1_run_manifest.bootstrap_indices.npz  

This file contains the exact bootstrap resampling indices used for AUROC and AUROC difference estimation.

The results file contains one entry per sample with labels and all computed scores.  
The manifest file records configuration, environment details, AUROC values, confidence intervals, and references to stored bootstrap indices.

---

## Sanity Checks

After a successful run:

- the results file contains 300 lines  
- the manifest includes AUROC entries for all evaluated methods  
- if present, the bootstrap index file can be used to exactly reproduce confidence intervals  

---

## Determinism

All sources of randomness are fixed:

- global seed  
- cross-validation splits  
- bootstrap resampling  

Repeated runs produce identical results, including bootstrap confidence intervals, up to floating-point precision.  
Minor floating-point differences may occur across different hardware architectures (e.g., CPU vs. GPU).

---

## Model Checkpoint

Phase-1 evaluation uses a single fixed model checkpoint for all frozen outputs.

Model parameters remain frozen throughout Phase 1.  
The exact checkpoint identifier is recorded in the run manifest.

---

## Notes

CPU-only execution is slower but yields equivalent results.  
No external services or APIs are required.