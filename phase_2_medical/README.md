# White-Box Hallucination Detection for Large Language Models

This repository contains the complete, reproducible implementation of Phase 2 of the master’s thesis
“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models”.

Phase 2 investigates whether white-box hallucination signals validated in Phase 1 generalize to
constrained medical question answering tasks.
In contrast to Phase 1, which focuses on free-form responses, Phase 2 evaluates hallucination detection
in multiple-choice and short-answer medical QA settings.

---

## Research Objective

The objective of Phase 2 is to study the robustness and transferability of white-box hallucination
signals across domains and task formats.

Hallucination detection is formulated as binary classification:

- error = 1: the model-selected answer is incorrect  
- error = 0: the model-selected answer is correct  

All methods are evaluated using AUROC as the primary performance metric.
Spearman rank correlation is reported as a complementary signal-quality measure.

---

## Methods and Signals

Phase 2 evaluates the probe-based and aggregated white-box scorers validated in Phase 1.

Implemented scorers include:

- LNTP (Length-Normalized Token Probability)  
- MTP (Minimum Token Probability)  
- EGH probe (out-of-fold)  
- hidden-state probe (out-of-fold)  

Individual EGH primitives are not evaluated separately in Phase 2, as Phase 1 already established
the superiority of supervised probe-based aggregation.

Logit-based scores are treated as ranking signals for error detection and not as calibrated
uncertainty estimates.

All supervised scorers are evaluated using leakage-safe out-of-fold predictions.

---

## Replication Scope

Phase 2 does not introduce new scoring mechanisms.
Instead, it evaluates whether the hallucination signals established in Phase 1 generalize to
medical question answering tasks under a unified evaluation protocol.

Not in scope for Phase 2:

- calibration or probabilistic uncertainty estimation  
- fine-tuning or domain adaptation  
- hallucination mitigation or intervention strategies  

---

## Datasets and Frozen Evaluation Setup

Evaluation is performed on frozen subsets of two medical QA benchmarks:

- MedQA (USMLE-style multiple-choice questions)  
- PubMedQA (biomedical yes/no questions)  

For each dataset:

- a fixed subset of 1,000 examples is selected  
- labels and model outputs are frozen prior to evaluation  
- no dataset-specific tuning is performed  

Frozen evaluation files are stored in:

benchmarks/medqa_test_labeled_seed42_n1000.jsonl  
benchmarks/pubmedqa_labeled_phase2.jsonl  

These files serve as the single source of truth for Phase 2.

---

## Models

Phase 2 evaluates two instruction-tuned 7B models:

- Mistral-7B-Instruct v0.2  
- BioMistral-7B  

Model outputs are generated once and reused unchanged across all scorers.
No generation-time sampling or uncertainty estimation is performed during evaluation.

---

## Evaluation Protocol

Phase 2 follows a conservative and fully reproducible evaluation protocol:

- fixed global random seed (42)  
- frozen datasets and model outputs  
- no score-specific filtering  
- AUROC as the primary metric  
- Spearman rank correlation as a secondary metric  
- stratified bootstrap confidence intervals (B = 5000)  

Bootstrap resampling indices are generated once per run and stored in the corresponding manifest
files to ensure exact reproducibility across all analyses.

---

## Analysis and Visualization

Final metrics, confidence intervals, tables, and figures are generated using deterministic scripts:

analysis/phase2_figures.py  
analysis/phase2_tables.py  

Primary visualizations include grouped bar plots comparing scorers across models,
with numeric value annotations for all reported metrics.

---

## Reproducibility

Phase 2 is fully reproducible using frozen inputs and deterministic scripts.

Run scoring:

python src/run_phase2.py  

Run analysis and figure generation:

python analysis/phase2_figures.py  
python analysis/phase2_tables.py  

All evaluation artifacts, including bootstrap indices, are written to disk for full auditability.

---

## Repository Structure

benchmarks/  
Frozen medical QA datasets and labels  

src/  
Data preparation and scoring pipeline  

outputs/  
Final results, manifests, bootstrap indices, and figures  

analysis/  
Deterministic evaluation, tables, and figure generation  

---

## Scope and Limitations

Phase 2 evaluates hallucination detection in constrained medical QA settings.
Hallucination is operationalized as incorrect answer selection and does not cover free-form
hallucination generation.

Token-based uncertainty measures may be disadvantaged in short-answer or multiple-choice formats.

Results are intended to assess generalization of hallucination signals rather than to establish
domain-specific performance claims.

---

## License and Usage

This repository is intended for academic and research use.
Users are responsible for complying with the licenses of external datasets and model checkpoints.