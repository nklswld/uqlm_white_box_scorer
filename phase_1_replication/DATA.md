# Phase 1 Data Documentation (`phase_1_replication/DATA.md`)

This repository contains the complete, reproducible implementation of Phase 1 of the master’s thesis  
“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models”.

Phase 1 investigates hallucination detection using white-box signals derived from a model’s internal computations.  
Uncertainty quantification in the strict probabilistic sense and calibration are addressed in later phases of the thesis.

Importantly, this repository implements a **controlled partial replication** of representative white-box hallucination scoring methods proposed in prior work. The focus lies on faithfully reproducing core signal definitions and extraction procedures under a unified experimental setup, rather than on reproducing full end-to-end systems, paper-specific training pipelines, or task-optimized detectors.

---

## Research Objective

The objective of Phase 1 is to study whether internal model signals can reliably discriminate hallucinated from non-hallucinated responses under controlled conditions. Rather than proposing a novel detection method, Phase 1 systematically compares representative white-box scoring approaches—namely logit-based confidence measures, hidden-state-based probes, and gradient-based discrepancy signals—within a unified and reproducible evaluation framework.

The task is formulated as binary classification:

- hallucinated = 1: the model answer contains at least one specific, falsifiable factually incorrect or fabricated claim  
- hallucinated = 0: the model answer does not contain such a claim and is factually correct or epistemically appropriate  

A single falsifiable incorrect claim suffices for a hallucination label (response-level annotation).

All methods are evaluated using AUROC as the primary performance metric.

---

## Scope and Limitations

Phase 1 evaluates hallucination detection under a deliberately restricted experimental setting. In particular, the replication omits paper-specific downstream classifier architectures, dataset-specific tuning strategies, and generation-time intervention mechanisms. This design choice prioritizes methodological isolation and comparability of white-box signals over task-optimized performance.

Evaluation is conducted on a frozen dataset with a single model checkpoint under teacher-forced conditions. Results therefore characterize signal behavior within this controlled configuration rather than across heterogeneous models, prompts, or decoding strategies.

Results are intended for methodological comparison of white-box signals and do not constitute leaderboard claims.

---

## License and Usage

This repository is intended for academic and research use.  
Users are responsible for complying with the licenses of external datasets and model checkpoints.