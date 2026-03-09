# White-Box Hallucination Detection and Uncertainty Scoring in LLMs

This repository contains the full implementation used in a master’s thesis:

**“White-Box Scores for Uncertainty Quantification and Hallucination Detection in Large Language Models.”**

The project is organized into two experimental phases:

- **Phase 1 (`phase_1_replication/`)**: controlled replication of white-box hallucination signals on TruthfulQA  
- **Phase 2 (`phase_2_medical/`)**: transfer evaluation of selected white-box signals on constrained medical QA (MedQA, PubMedQA)

---

## What This Repository Is (and Is Not)

This is a **research artifact repository** focused on:

- post-hoc white-box scoring
- leakage-safe comparative evaluation
- reproducible reporting from frozen artifacts

It is **not** intended as:

- a production software package
- an API/service deployment project
- a generic end-user application

---

## Project Scope

In scope:

- logit-based token confidence signals (e.g., LNTP, MTP)
- gradient-based internal discrepancy signals
- hidden-state probe-based scoring
- out-of-fold evaluation for supervised probes
- AUROC/Spearman reporting with bootstrap confidence intervals

Out of scope:

- model fine-tuning
- calibration pipelines for probabilistic uncertainty estimates
- generation-time mitigation/intervention methods

---

## Read This First

For a fast and correct understanding order:

1. `ARTIFACT_MANIFEST.md`
2. `benchmarks_overview.txt`
3. `phase_1_replication/README.md`
4. `phase_1_replication/reproduce_phase1.md`
5. `phase_2_medical/README.md`
6. `phase_2_medical/reproduce_phase2.md`
7. `phase_2_medical/DATA.md`

---

## Reproduction Strategy (Important)

For this project, a **full end-to-end rerun is not the default path** because it is computationally expensive.

Use the following tiers:

## Tier A — Reported Artifacts (Recommended Default)

Goal: verify and inspect the reported results using archived outputs.

- Use already archived outputs under:
  - `phase_1_replication/outputs/`
  - `phase_2_medical/outputs/final/`
  - `phase_2_medical/outputs/figures_tables/`

This is the preferred path for reviewers/examiners.

## Tier B — Partial Reproduction

Goal: regenerate tables/figures from existing results/manifests.

Commands:

```bash
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

(Phase-1 figure regeneration is available via `phase_1_replication/analysis/phase1_figures.py`.)

## Tier C — Full Rerun (Compute Intensive)

Goal: rerun scoring pipelines from frozen inputs or regenerate frozen predictions.

Typical entrypoints:

```bash
python phase_1_replication/src/run_phase1_truthfulqa.py
bash phase_2_medical/scripts/run_baseline_all.sh
```

Use this only when scientifically necessary.

---

## Canonical Artifacts / Source of Truth

Canonical artifacts for interpretation and reporting are documented in:

- `ARTIFACT_MANIFEST.md`
- `phase_2_medical/DATA.md`
- `phase_2_medical/reproduce_phase2.md`

For Phase 2 baseline reporting, canonical file families are:

- `*.results.jsonl`
- `*.manifest.json`
- `*.manifest.bootstrap_indices.npz`

Note: `.npz` bootstrap index files may be excluded from Git tracking due size policy (`.gitignore`).  
If exact CI index replay is required, use the provided artifact bundle/local archive.

---

## Repository Structure

- `phase_1_replication/`  
  Phase-1 code, frozen TruthfulQA artifacts, reproducibility docs

- `phase_2_medical/`  
  Phase-2 medical QA pipeline, frozen predictions, analysis scripts, ablations

- `benchmarks_overview.txt`  
  Benchmark-level overview across phases

- `ARTIFACT_MANIFEST.md`  
  Artifact inventory and canonical/optional classification

- `requirements.txt`  
  Shared Python dependencies

- `CITATION.cff`  
  Citation metadata for this repository

---

## Environment and Dependencies

Install dependencies from repository root:

```bash
pip install -r requirements.txt
```

General notes:

- Python `>= 3.10`
- GPU is recommended for heavy extraction/scoring workloads
- CPU execution is possible but slower
- Hugging Face access/model availability can affect full reruns

---

## Validated Environment

The repository was validated in a research setting with:

- Python >= 3.10
- dependencies from `requirements.txt`
- GPU-backed execution for compute-intensive scoring steps

Reference hardware used for main experiment runs:

- NVIDIA **A100 SXM (80 GB VRAM)** GPU via **RunPod**
- ~16 vCPU / ~117 GB RAM environment

Important notes:

- Exact bitwise reproducibility across different hardware/driver stacks is not guaranteed.
- Small floating-point deviations may occur across CUDA/PyTorch/runtime combinations.
- Reported baseline artifacts remain the reference for thesis-level result verification.

---

## Practical Reproducibility Notes

This repository is designed primarily as a research artifact archive, not as a lightweight rerun package.

- Recreating reported tables/figures from archived outputs is straightforward.
- Re-running selected experiments is possible with the provided scripts.
- A full end-to-end rerun is computationally expensive and is not the default verification path.
- Depending on hardware, model access, and environment, a complete rerun can take many hours.

---

## Compute and Runtime Note

- **Artifact-first verification is recommended**: validate reported results from archived outputs before attempting reruns.
- **Partial analysis regeneration** (tables/figures from archived artifacts) is typically quick on standard CPU setups.
- **Full end-to-end reruns are compute-intensive** and may take several hours (often up to ~10+ hours), depending on GPU/VM, model access, and I/O performance.
- For heavy scoring stages, a GPU-backed environment is strongly recommended.

---

## Non-Versioned Large Artifacts

Some large binary artifacts (e.g. certain `.npz` files such as serialized bootstrap indices) are intentionally not tracked in Git for storage reasons.

Implication:
- The repository contains the code and documented artifact structure,
  but some large intermediate or audit files may not be present in a fresh clone.

If needed for review or audit, these files can be provided separately or regenerated where supported by the documented pipeline.

---

## External Data / Model Access

Some workflows depend on external model and dataset access (Hugging Face).  
Users are responsible for:

- valid access permissions/tokens where required
- compliance with dataset/model licenses and terms

---

## Citation

Please cite this repository using metadata in:

- `CITATION.cff`

---

## License and Usage

This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

- You may use, share, and adapt the material for academic and other non-commercial purposes with proper attribution.
- Commercial use is not permitted without separate permission.

License:
- `LICENSE`
- https://creativecommons.org/licenses/by-nc/4.0/

Third-party notice:
External datasets, model checkpoints, and related resources referenced by this project remain subject to their own licenses and terms of use.