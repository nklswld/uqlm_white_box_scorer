# ARTIFACT_MANIFEST.md

This document inventories the research artifacts in this repository and defines which files are treated as canonical for reporting and reproduction.

Scope:
- Repository: `uqlm_white_box_scorer`
- Phases: `phase_1_replication`, `phase_2_medical`
- Focus: scientific reproducibility and auditability (not software packaging)

---

## 1) Artifact Policy (Canonical vs Optional)

## Canonical for reported results

These files are the primary source of truth for thesis-relevant reported outputs:

- Phase 1:
  - `phase_1_replication/outputs/phase1_truthfulqa_hallu_results_300.jsonl`
  - `phase_1_replication/outputs/phase1_run_manifest.json`
  - `phase_1_replication/outputs/self_iaa_summary.json`
- Phase 2 baseline:
  - `phase_2_medical/outputs/final/*.results.jsonl`
  - `phase_2_medical/outputs/final/*.manifest.json`
  - `phase_2_medical/outputs/final/*.manifest.bootstrap_indices.npz` (see tracking note below)

## Input anchors (frozen/prepared data)

- Phase 1 frozen benchmark:
  - `phase_1_replication/benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`
- Phase 2 prepared benchmark subsets:
  - `phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl`
  - `phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl`
- Phase 2 frozen prediction inputs:
  - `phase_2_medical/outputs/frozen/medqa_mistral7b.jsonl`
  - `phase_2_medical/outputs/frozen/medqa_biomistral7b.jsonl`
  - `phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl`
  - `phase_2_medical/outputs/frozen/pubmedqa_biomistral7b.jsonl`

## Optional / supplementary artifacts

- Phase 1 figures: `phase_1_replication/outputs/figs/*.pdf`
- Phase 2 analysis exports:
  - `phase_2_medical/outputs/figures_tables/tables_general/**`
  - `phase_2_medical/outputs/figures_tables/figures_general/**`
- Phase 2 ablations:
  - `phase_2_medical/outputs/ablations/**`
  - `phase_2_medical/outputs/figures_tables/ablations/**`

---

## 2) Inventory (Current Repository State)

## Phase 1

### Benchmarks / labels
- `phase_1_replication/benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl` (300 rows)
- `phase_1_replication/benchmarks/truthfulqa_simplified.json`
- `phase_1_replication/benchmarks/self_iaa_round2_labels_seed42_n80.jsonl`
- `phase_1_replication/benchmarks/self_iaa_round2_qids_seed42_n80.json`
- `phase_1_replication/benchmarks/labeling_guidelines_hallucination.md`

### Final outputs
- `phase_1_replication/outputs/phase1_truthfulqa_hallu_results_300.jsonl`
- `phase_1_replication/outputs/phase1_run_manifest.json`
  - Note: this manifest is relatively large because bootstrap samples and index matrices are serialized directly in JSON. The corresponding bootstrap indices are also stored in compressed form in `phase1_run_manifest.bootstrap_indices.npz`.
- `phase_1_replication/outputs/self_iaa_summary.json`
- `phase_1_replication/outputs/figs/*.pdf` (figure set)

### Bootstrap indices
- No tracked Phase-1 `.npz` currently present in Git.

---

## Phase 2

### Prepared benchmark subsets
- `phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl` (1000 rows)
- `phase_2_medical/benchmarks/medqa_test_qids_seed42_n1000.json`
- `phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl` (1000 rows)

### Frozen prediction inputs
- `phase_2_medical/outputs/frozen/medqa_mistral7b.jsonl` (1000 rows)
- `phase_2_medical/outputs/frozen/medqa_biomistral7b.jsonl` (1000 rows)
- `phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl` (1000 rows)
- `phase_2_medical/outputs/frozen/pubmedqa_biomistral7b.jsonl` (1000 rows)
- `phase_2_medical/outputs/frozen/truthfulqa_hallu_mistral_like.jsonl` (used for token-bias appendix/ablation flow)

### Baseline final outputs (`outputs/final`)
- `medqa_mistral.B5000.results.jsonl` (1000 rows)
- `medqa_mistral.B5000.manifest.json`
- `medqa_mistral.B5000.manifest.bootstrap_indices.npz`
- `medqa_biomistral.B5000.results.jsonl` (1000 rows)
- `medqa_biomistral.B5000.manifest.json`
- `medqa_biomistral.B5000.manifest.bootstrap_indices.npz`
- `pubmedqa_mistral.B5000.results.jsonl` (1000 rows)
- `pubmedqa_mistral.B5000.manifest.json`
- `pubmedqa_mistral.B5000.manifest.bootstrap_indices.npz`
- `pubmedqa_biomistral.B5000.results.jsonl` (1000 rows)
- `pubmedqa_biomistral.B5000.manifest.json`
- `pubmedqa_biomistral.B5000.manifest.bootstrap_indices.npz`

### Supplementary outputs
- Analysis tables/figures:
  - `phase_2_medical/outputs/figures_tables/tables_general/**`
  - `phase_2_medical/outputs/figures_tables/figures_general/**`
- Ablation outputs and analysis:
  - `phase_2_medical/outputs/ablations/**`
  - `phase_2_medical/outputs/figures_tables/ablations/**`

---

## 3) Tracking and Size Policy

`.gitignore` currently includes:

- `*.npz` (intentionally ignored due artifact size constraints)

Implication:
- Bootstrap index files (`*.manifest.bootstrap_indices.npz`) may exist locally and be used in analysis, but are not guaranteed to be present in a fresh Git clone.
- If exact CI regeneration from archived resampling indices is required, these `.npz` files must be transferred out-of-band (e.g., archive bundle).

---

## 4) Reproduction Levels

## Level A: Reported-artifact reproduction (recommended)
Use existing archived outputs:
- Phase 1: `phase_1_replication/outputs/*`
- Phase 2 baseline: `phase_2_medical/outputs/final/*` (+ `.npz` if available)

## Level B: Partial recomputation
Recompute tables/figures from existing results/manifests:
- `phase_1_replication/analysis/phase1_figures.py`
- `phase_2_medical/analysis/phase2_tables.py`
- `phase_2_medical/analysis/phase2_figures.py`

## Level C: Full rerun (compute intensive)
Regenerate frozen outputs and rerun scoring pipelines:
- Phase 1: `phase_1_replication/src/run_phase1_truthfulqa.py`
- Phase 2: `phase_2_medical/scripts/run_baseline_all.sh` (plus optional ablations)

See:
- `phase_1_replication/reproduce_phase1.md`
- `phase_2_medical/reproduce_phase2.md`

---

## 5) Minimal Audit Checklist

1. Confirm benchmark anchors exist:
   - `phase_1_replication/benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`
   - `phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl`
   - `phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl`
2. Confirm Phase-2 frozen inputs exist (4 files under `phase_2_medical/outputs/frozen/`).
3. Confirm each baseline run has:
   - `.results.jsonl`
   - `.manifest.json`
   - `.manifest.bootstrap_indices.npz` (if provided externally, due ignore policy)
4. Confirm table exports exist:
   - `phase_2_medical/outputs/figures_tables/tables_general/phase2_metrics_auroc_ci.csv`
   - `phase_2_medical/outputs/figures_tables/tables_general/phase2_metrics_spearman_rho.csv`
5. Confirm figure exports exist:
   - files under `phase_2_medical/outputs/figures_tables/figures_general/`

---

## 6) License and Citation Metadata

Repository-level metadata files:

- `LICENSE` (or `LICENSE.md`):  
  Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

- `CITATION.cff`:  
  Canonical citation metadata for software/research artifact referencing.

Scope note:

- The repository license applies to original code and documentation in this project.
- External datasets, model checkpoints, and other third-party resources remain subject to their own licenses and terms.

Reference links:

- https://creativecommons.org/licenses/by-nc/4.0/
- https://creativecommons.org/licenses/by-nc/4.0/legalcode

---

## 7) Related Documentation

- Root overview: `README.md`
- Benchmark summary: `benchmarks_overview.md`
- Phase 1:
  - `phase_1_replication/README.md`
  - `phase_1_replication/DATA.md`
  - `phase_1_replication/reproduce_phase1.md`
- Phase 2:
  - `phase_2_medical/README.md`
  - `phase_2_medical/DATA.md`
  - `phase_2_medical/reproduce_phase2.md`