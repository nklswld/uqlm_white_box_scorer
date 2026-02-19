# reproduce_phase2.md

## Purpose

This document describes how to reproduce all Phase-2 results in this repository.

Phase 2 evaluates the generalization of white-box hallucination detection signals to constrained
medical question answering tasks.
All experiments operate exclusively on frozen model outputs and are fully deterministic.

No text generation, sampling, fine-tuning, or labeling is performed during reproduction.

---

## Frozen Outputs

Phase-2 experiments rely exclusively on frozen model output files generated once and never modified.

All white-box scores are computed from these frozen outputs.
No score-specific filtering or resampling is performed.

All Phase-2 computations use teacher forcing only.

---

## Datasets

Required files:

benchmarks/medqa_test_labeled_seed42_n1000.jsonl  
benchmarks/pubmedqa_labeled_phase2.jsonl  

Each line in the dataset files is a JSON object containing:

- qid: unique question identifier  
- question: medical question text  
- answer options or context (task-dependent)  
- model_answer: model-selected answer  
- error: binary correctness label (0 = correct, 1 = incorrect)  

No other datasets are required.

---

## Environment

- Python >= 3.10  
- Dependencies installed via:  
  pip install -r requirements.txt  

Tested with GPU execution.
GPU is required for gradient- and hidden-state-based white-box score extraction.

---

## Reproduction

Verify repository structure:

phase_2_medical/  
- src/  
- benchmarks/  
- outputs/  
- analysis/  

Phase 2 results are produced by running the scoring script on frozen model outputs.
Each run generates one results file and one manifest file.

The following commands reproduce the final Phase-2 outputs used for all analyses and figures.

---

### Mistral × PubMedQA

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python phase_2_medical/src/run_phase2.py \
  --task pubmedqa \
  --frozen_jsonl phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl \
  --out_jsonl phase_2_medical/outputs/final/pubmedqa_mistral.B5000.results.jsonl \
  --out_manifest phase_2_medical/outputs/final/pubmedqa_mistral.B5000.manifest.json \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --B 5000 \
  --batch_size 4 \
  --hidden_batch_size 4 \
  --max_context_tokens 128
