# Benchmarks Overview

This document summarizes the benchmark datasets used across both experimental phases, their role in the repository, and the corresponding local artifacts.

In Phase 2, these benchmarks are used in a constrained prediction-error setting rather than as a generic free-form hallucination benchmark.

---

## 1) TruthfulQA (Phase 1 core benchmark)

- Source:
  - https://huggingface.co/datasets/truthful_qa
- Repository role:
  - Main benchmark for the Phase-1 conceptual replication (hallucinated vs non-hallucinated responses under a controlled post-hoc setting).
- Canonical local artifacts:
  - phase_1_replication/benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl
  - phase_1_replication/benchmarks/truthfulqa_simplified.json
- Notes:
  - Phase 1 evaluation runs on frozen model outputs (post-hoc scoring).
  - A derived TruthfulQA-shaped file is also used in one Phase-2 ablation path:
    - phase_2_medical/outputs/frozen/truthfulqa_hallu_mistral_like.jsonl

---

## 2) MedQA-USMLE-4-options (Phase 2 medical benchmark)

- Source:
  - https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options
- Repository role:
  - Phase 2 constrained medical multiple-choice benchmark.
- Preparation script:
  - phase_2_medical/src/prepare_medqa_phase2.py
- Canonical prepared artifact:
  - phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl
  - phase_2_medical/benchmarks/medqa_test_qids_seed42_n1000.json
- Notes:
  - Seeded, without-replacement subsample from MedQA test split (default n=1000, seed=42).
  - Used to generate frozen prediction files for Phase 2 scoring.

---

## 3) PubMedQA (pqa_labeled) (Phase 2 medical benchmark)

- Source:
  - https://huggingface.co/datasets/qiaojin/PubMedQA
- Repository role:
  - Phase 2 constrained medical yes/no/maybe benchmark.
- Preparation script:
  - phase_2_medical/src/prepare_pubmedqa_phase2.py
- Canonical prepared artifact:
  - phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl
- Notes:
  - Upstream source in the preparation script: `pqa_labeled` train split.
  - Released repository artifact: `pubmedqa_labeled_phase2.jsonl`, the prepared 1,000-example PubMedQA subset used for frozen generation and Phase-2 evaluation.
  - In thesis-level summaries, Phase 2 may be described more generically as operating on frozen 1,000-example evaluation subsets across datasets; for PubMedQA, interpret that wording as referring to the released prepared subset rather than the upstream split name.
  - Gold labels are normalized to {yes, no, maybe}.

---

## Frozen prediction artifacts (not raw benchmarks, but evaluation inputs)

Phase 2 baseline scoring consumes frozen model outputs from:

- phase_2_medical/outputs/frozen/medqa_mistral7b.jsonl
- phase_2_medical/outputs/frozen/medqa_biomistral7b.jsonl
- phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl
- phase_2_medical/outputs/frozen/pubmedqa_biomistral7b.jsonl

These are generated with:

- phase_2_medical/src/generate_frozen_phase2.py

---

## Canonical result artifacts for reported numbers

For thesis-level reported Phase-2 baseline numbers, canonical outputs are:

- phase_2_medical/outputs/final/*.results.jsonl
- phase_2_medical/outputs/final/*.manifest.json
- phase_2_medical/outputs/final/*.manifest.bootstrap_indices.npz

---

## Practical data location note

When running preparation scripts, Hugging Face datasets are downloaded to the local HF cache (platform-dependent; typically under the user cache directory). Prepared benchmark files and frozen artifacts in this repository should be treated as the reproducibility anchors for this project.
