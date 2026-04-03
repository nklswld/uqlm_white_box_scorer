# White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models

This repository contains the implementation and archived outputs used in a master's thesis:

**"White-Box Scorers for Uncertainty Quantification and Hallucination Detection in Large Language Models."**

The project is organized into two experimental phases:

- **Phase 1 (`phase_1_replication/`)**: conceptual replication of representative white-box hallucination scoring approaches on TruthfulQA under a controlled post-hoc setting
- **Phase 2 (`phase_2_medical/`)**: transfer evaluation of selected white-box signals for benchmark-defined prediction error detection on constrained medical QA (MedQA, PubMedQA)

---

## What This Repository Is (and Is Not)

This is a **research artifact repository** focused on:

- post-hoc white-box scoring and supervised white-box readouts
- leakage-safe comparative evaluation
- reproducible reporting from frozen artifacts

It is **not** intended as:

- a production software package
- an API/service deployment project
- a generic end-user application

---

## Project Scope

In scope:

- intrinsic logit-based token confidence scores (e.g., LNTP, MTP)
- supervised white-box probes based on hidden states or gradient/embedding features
- out-of-fold evaluation for supervised probes
- AUROC/Spearman reporting for ranking-based discrimination with bootstrap confidence intervals

Out of scope:

- model fine-tuning
- calibration pipelines for probabilistic uncertainty estimates
- generation-time mitigation/intervention methods

---

## Reading Order

For orientation, read the documentation in this order:

1. `ARTIFACT_MANIFEST.md`
2. `benchmarks_overview.md`
3. `phase_1_replication/README.md`
4. `phase_1_replication/DATA.md`
5. `phase_1_replication/reproduce_phase1.md`
6. `phase_2_medical/README.md`
7. `phase_2_medical/DATA.md`
8. `phase_2_medical/reproduce_phase2.md`

---

## Reproduction Levels

For this repository, full end-to-end reruns are possible but are not the default verification path.

### Tier A - Archived Artifacts

Use the archived outputs directly:

- `phase_1_replication/outputs/`
- `phase_2_medical/outputs/final/`
- `phase_2_medical/outputs/figures_tables/`

This is the preferred path for thesis review, result inspection, and artifact audit.

### Tier B - Analysis Regeneration

Regenerate derived tables and figures from archived results and manifests:

```bash
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

Phase-1 figure regeneration is available via `phase_1_replication/analysis/phase1_figures.py`.

### Tier C - Pipeline Rerun

Rerun scoring from frozen inputs or regenerate frozen predictions when a full procedural check is required:

```bash
python phase_1_replication/src/run_phase1_truthfulqa.py
bash phase_2_medical/scripts/run_baseline_all.sh
```

This tier is compute-intensive and environment-dependent.

---

## Canonical Artifacts

Canonical artifacts for interpretation and reporting are documented in:

- `ARTIFACT_MANIFEST.md`
- `phase_1_replication/DATA.md`
- `phase_1_replication/reproduce_phase1.md`
- `phase_2_medical/DATA.md`
- `phase_2_medical/reproduce_phase2.md`

For Phase 2 baseline reporting, canonical file families are:

- `*.results.jsonl`
- `*.manifest.json`
- `*.manifest.bootstrap_indices.npz`

Current repository state:

- Phase 2 baseline bootstrap index files under `phase_2_medical/outputs/final/` are present in this repository snapshot.
- Phase 1 bootstrap index files are not tracked in Git.

Additional locally generated `.npz` artifacts are not guaranteed to be tracked because they remain covered by the repository size policy (`.gitignore`).

Manifest note:

- archived manifest files may preserve absolute paths from the original execution environment
- these path fields support auditability and are not expected to resolve unchanged on every machine

---

## Repository Structure

- `phase_1_replication/`
  Phase-1 code, frozen TruthfulQA artifacts, reproducibility docs

- `phase_2_medical/`
  Phase-2 medical QA pipeline, frozen predictions, analysis scripts, ablations

- `benchmarks_overview.md`
  Benchmark-level overview across phases

- `ARTIFACT_MANIFEST.md`
  Artifact inventory and canonical/optional classification

- `requirements.txt`
  Shared Python dependencies

- `CITATION.cff`
  Citation metadata for this repository

---

## Environment and Runtime

Install dependencies from repository root:

```bash
pip install -r requirements.txt
```

General environment notes:

- Python `>= 3.10`
- GPU is recommended for heavy extraction/scoring workloads
- CPU execution is possible but slower
- Hugging Face access/model availability can affect full reruns
- maintained `bash` runners in `phase_2_medical/scripts/` assume a Linux-like shell environment; Linux or WSL are the intended targets for those entrypoints

Validated reference environment:

- Python >= 3.10
- dependencies from `requirements.txt`
- GPU-backed execution for compute-intensive scoring steps

Reference hardware used for main experiment runs:

- NVIDIA **A100 SXM (80 GB VRAM)** GPU via **RunPod**
- ~16 vCPU / ~117 GB RAM environment

Runtime notes:

- Exact bitwise reproducibility across different hardware/driver stacks is not guaranteed.
- Small floating-point deviations may occur across CUDA/PyTorch/runtime combinations.
- Reported baseline artifacts remain the reference for thesis-level result verification.

Practical reproducibility note:

- artifact-first verification is recommended before attempting reruns
- partial analysis regeneration is lightweight relative to full scoring reruns
- complete reruns can take many hours depending on GPU availability, model access latency, and I/O performance
- on the reference environment, heavy reruns typically fall in the ~6-10 hour range

---

## Large Binary Artifacts

Some large binary artifacts (e.g. certain `.npz` files such as serialized bootstrap indices) are intentionally not tracked in Git for storage reasons.

Repository-state note:

- Phase 1 bootstrap index files are not tracked in Git.
- The four Phase 2 baseline bootstrap index files under `phase_2_medical/outputs/final/` are present in the current repository state.

Additional large intermediate or audit files may still be absent from a fresh clone. Where needed for review or archival verification, they are available in the accompanying submission archive or can be regenerated where the documented pipeline supports it.

---

## External Data / Model Access

Some workflows depend on external model and dataset access (Hugging Face).
Users are responsible for:

- valid access permissions/tokens where required
- compliance with dataset/model licenses and terms

---

## Citation and License

Citation metadata:

- `CITATION.cff`

This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- You may use, share, and adapt the material for academic and other non-commercial purposes with proper attribution.
- Commercial use is not permitted without separate permission.

License references:
- `LICENSE`
- https://creativecommons.org/licenses/by-nc/4.0/

Third-party notice:
External datasets, model checkpoints, and related resources referenced by this project remain subject to their own licenses and terms of use.
