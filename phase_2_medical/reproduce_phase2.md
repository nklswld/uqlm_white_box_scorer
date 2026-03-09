# reproduce_phase2.md

## Purpose

This document provides the exact procedure to reproduce all Phase-2 medical QA scoring artifacts, metrics, and downstream analysis outputs from this repository.

Phase 2 evaluates whether white-box uncertainty/hallucination signals from Phase 1 generalize to constrained medical QA tasks under a frozen-output protocol.

---

## Reproduction Scope

Phase 2 reproduction includes:

- baseline scorer runs (MedQA/PubMedQA × Mistral/BioMistral)
- manifests and bootstrap index artifacts
- deterministic table/figure generation
- optional ablation runs

Phase 2 reproduction does **not** include:

- model fine-tuning
- calibration pipelines
- generation-time mitigation/intervention methods

---

## Canonical Artifact Rule

For reported baseline results, canonical artifacts are:

- `phase_2_medical/outputs/final/*.results.jsonl`
- `phase_2_medical/outputs/final/*.manifest.json`
- `phase_2_medical/outputs/final/*.manifest.bootstrap_indices.npz`

If these files already exist and match expected naming, they are the source of truth for analysis reproduction.

---

## Required Inputs

### Required for baseline scoring and analysis

Frozen prediction files:

- `phase_2_medical/outputs/frozen/medqa_mistral7b.jsonl`
- `phase_2_medical/outputs/frozen/medqa_biomistral7b.jsonl`
- `phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl`
- `phase_2_medical/outputs/frozen/pubmedqa_biomistral7b.jsonl`

### Required only if regenerating frozen outputs

Prepared benchmark files:

- `phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl`
- `phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl`

---

## Environment

Minimum:

- Python `>= 3.10`
- Dependencies from repo root:

```bash
pip install -r requirements.txt
```

Execution assumptions:

- GPU is recommended/expected for gradient + hidden feature extraction.
- Hugging Face access token may be required depending on model availability.

Optional `.env` in repo root:

```bash
HF_TOKEN=...
```

Some runner scripts map `HF_TOKEN` to `HUGGINGFACE_HUB_TOKEN`.

---

## Determinism Notes

The pipeline is designed for high reproducibility via:

- frozen input predictions
- fixed seeds in baseline configurations
- serialized bootstrap indices per run
- deterministic analysis scripts over archived artifacts

Practical note:

- very small floating-point deviations may occur across runtime/library/hardware environments.
- canonical reported outputs are defined by archived files under `outputs/final/`.

---

## Baseline Reproduction (Recommended Path)

Run all 4 baseline conditions with the maintained script:

```bash
bash phase_2_medical/scripts/run_baseline_all.sh
```

This script executes:

1. Mistral × PubMedQA
2. Mistral × MedQA
3. BioMistral × PubMedQA
4. BioMistral × MedQA

with fixed baseline settings (including task-specific token budgets).

Expected output files:

- `phase_2_medical/outputs/final/pubmedqa_mistral.B5000.results.jsonl`
- `phase_2_medical/outputs/final/pubmedqa_mistral.B5000.manifest.json`
- `phase_2_medical/outputs/final/pubmedqa_mistral.B5000.manifest.bootstrap_indices.npz`
- `phase_2_medical/outputs/final/medqa_mistral.B5000.results.jsonl`
- `phase_2_medical/outputs/final/medqa_mistral.B5000.manifest.json`
- `phase_2_medical/outputs/final/medqa_mistral.B5000.manifest.bootstrap_indices.npz`
- `phase_2_medical/outputs/final/pubmedqa_biomistral.B5000.results.jsonl`
- `phase_2_medical/outputs/final/pubmedqa_biomistral.B5000.manifest.json`
- `phase_2_medical/outputs/final/pubmedqa_biomistral.B5000.manifest.bootstrap_indices.npz`
- `phase_2_medical/outputs/final/medqa_biomistral.B5000.results.jsonl`
- `phase_2_medical/outputs/final/medqa_biomistral.B5000.manifest.json`
- `phase_2_medical/outputs/final/medqa_biomistral.B5000.manifest.bootstrap_indices.npz`

---

## Direct CLI Reproduction (Single Run Example)

If you want one run directly via Python:

```bash
python phase_2_medical/src/run_phase2.py \
  --task pubmedqa \
  --frozen_jsonl phase_2_medical/outputs/frozen/pubmedqa_mistral7b.jsonl \
  --out_jsonl phase_2_medical/outputs/final/pubmedqa_mistral.B5000.results.jsonl \
  --out_manifest phase_2_medical/outputs/final/pubmedqa_mistral.B5000.manifest.json \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --device cuda:0 \
  --dtype bfloat16 \
  --seed 42 \
  --n_splits 5 \
  --B 5000 \
  --ci 0.95 \
  --batch_size 4 \
  --hidden_batch_size 4 \
  --max_context_tokens 128
```

For baseline comparability, prefer the script values in `run_baseline_all.sh`.

---

## Analysis Reproduction

After baseline outputs are present:

```bash
python phase_2_medical/analysis/phase2_tables.py
python phase_2_medical/analysis/phase2_figures.py
```

Expected output roots:

- Tables: `phase_2_medical/outputs/figures_tables/tables_general/`
- Figures: `phase_2_medical/outputs/figures_tables/figures_general/`

---

## Optional Ablation Reproduction

Available ablation runners:

- `phase_2_medical/scripts/run_ablation_oof_seeds.sh`
- `phase_2_medical/scripts/run_ablation_n_splits.sh`
- `phase_2_medical/scripts/run_ablation_bootstrap_budget.sh`
- `phase_2_medical/scripts/run_ablation_hidden_layers.sh`
- `phase_2_medical/scripts/run_ablation_hidden_pooling.sh`
- `phase_2_medical/scripts/run_ablation_token_score_bias.sh`

Run example:

```bash
bash phase_2_medical/scripts/run_ablation_oof_seeds.sh
```

Ablation outputs are written under `phase_2_medical/outputs/ablations/`.

Note:

- `run_ablation_token_score_bias.sh` uses the frozen TruthfulQA artifact (`outputs/frozen/truthfulqa_hallu_mistral_like.jsonl`) for methodological appendix analysis, not the medical baseline benchmark pair.

---

## Validation Checklist

After reproduction, validate:

1. each baseline run has `results + manifest + bootstrap_indices`
2. manifest `task/model/config` matches intended run
3. `hidden_coverage.coverage` is reported (baseline should be full coverage in released artifacts)
4. final metric CSVs are generated:
   - `phase2_metrics_auroc_ci.csv`
   - `phase2_metrics_spearman_rho.csv`
5. expected figure/table PDFs/CSVs exist in `outputs/figures_tables/`

---

## Common Pitfalls

- Missing HF auth for model loading
- Running from wrong working directory
- Overwriting previously archived outputs unintentionally
- Mixing environments (library versions) and expecting bit-identical probe scores

---

## Data and Schema References

For benchmark/frozen/result schema details, see:

- `phase_2_medical/DATA.md`

For high-level project overview, see:

- `phase_2_medical/README.md`