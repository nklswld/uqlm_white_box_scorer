# White-Box Hallucination Detection for Large Language Models

This repository contains the complete, reproducible implementation of Phase 2 of the master’s thesis  
“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models”.

Phase 2 investigates whether white-box hallucination detection signals validated in Phase 1 generalize to
constrained medical question answering tasks. In contrast to Phase 1, which focuses on free-form responses,
Phase 2 evaluates hallucination detection in multiple-choice and short-answer medical QA settings.

Importantly, Phase 2 does not introduce new scoring methods. Instead, it evaluates the robustness and
transferability of previously validated white-box signals under a different task structure and domain.

---

## Research Objective

The objective of Phase 2 is to assess whether white-box hallucination detection signals derived from a model’s
internal computations remain informative when applied to constrained medical question answering tasks.

Rather than proposing new detection mechanisms, Phase 2 examines the generalization of logit-based and
probe-based white-box scorers across:

- different domains (general knowledge → medical)
- different task formats (free-form → multiple-choice / short answers)
- different model checkpoints

Hallucination detection is formulated as binary classification:

- error = 1: the model-selected answer is incorrect  
- error = 0: the model-selected answer is correct  

AUROC is used as the primary evaluation metric.

---

## Datasets and Evaluation Setup

Phase 2 evaluates hallucination detection on two medical question answering benchmarks:

- **MedQA**, consisting of USMLE-style multiple-choice medical questions  
- **PubMedQA**, consisting of biomedical yes/no questions derived from scientific abstracts  

For both datasets, hallucination detection is operationalized as incorrect answer selection.

A fixed subset of 1,000 examples is selected from each dataset using a global random seed (42).
All datasets are evaluated under a strictly frozen setup: dataset splits, labels, and model outputs are
fixed prior to evaluation and reused unchanged across all scorers.

The frozen evaluation files are stored in:

benchmarks/medqa_test_labeled_seed42_n1000.jsonl  
benchmarks/pubmedqa_labeled_phase2.jsonl  

These files serve as the single source of truth for all Phase-2 evaluations.

---

## Definition of Hallucination in Phase 2

In Phase 2, hallucination detection is operationalized as incorrect answer selection in constrained medical
question answering tasks.

This definition differs from Phase 1, which focuses on free-form hallucination generation.
Phase 2 therefore evaluates whether hallucination signals generalize to settings where outputs are limited
to predefined answer options or short responses.

---

## Scope and Limitations

Phase 2 evaluates hallucination detection under a deliberately constrained experimental setting. In particular,
the evaluation omits free-form text generation, domain-specific fine-tuning, and generation-time intervention
mechanisms.

Token-based uncertainty measures may be disadvantaged in short-answer or multiple-choice formats, and medical
correctness is treated as a binary, task-specific signal. Results are intended to assess generalization of
hallucination detection signals rather than to establish domain-specific performance claims.

---

## License and Usage

This repository is intended for academic and research use.  
Users are responsible for complying with the licenses of external datasets and model checkpoints.