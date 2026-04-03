# Phase 2 Data Documentation (`phase_2_medical/DATA.md`)

This document specifies the Phase-2 data artifacts, schemas, and labeling conventions used in the medical QA evaluation pipeline.

---

## Scope

Phase 2 is evaluated in a **frozen-output, post-hoc** setting:

- no generation during scorer evaluation
- no relabeling during scorer evaluation
- no dataset-specific tuning during scorer evaluation

Reported metrics are anchored to archived artifacts under `phase_2_medical/outputs/`.

Phase 2 evaluates **benchmark-defined prediction error** under constrained medical QA. This preserves the binary detection form of Phase 1, but not the exact Phase-1 hallucination construct.

---

## Canonical Data Principle

For reported Phase-2 results, canonical artifacts are:

- `phase_2_medical/outputs/final/*.results.jsonl`
- `phase_2_medical/outputs/final/*.manifest.json`
- `phase_2_medical/outputs/final/*.manifest.bootstrap_indices.npz`

Preparation scripts remain available for transparency, but reproducibility of reported numbers is defined against these archived files.

---

## Directory Map

- `phase_2_medical/benchmarks/`  
  Prepared benchmark subsets and labels

- `phase_2_medical/outputs/frozen/`  
  Frozen model predictions used as scoring input

- `phase_2_medical/outputs/final/`  
  Final scorer outputs and manifests

- `phase_2_medical/outputs/ablations/`  
  Ablation-specific result artifacts

- `phase_2_medical/outputs/figures_tables/`  
  Derived summary tables and figures built from the archived result artifacts

---

## Benchmark Files

### 1) MedQA prepared subset

Files:

- `phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl`
- `phase_2_medical/benchmarks/medqa_test_qids_seed42_n1000.json`

Purpose:

- fixed 1,000-example MedQA test subset for Phase 2
- deterministic subset membership via seed (`42`)

JSONL schema (`medqa_test_labeled_seed42_n1000.jsonl`):

```json
{
  "qid": "medqa::test::000123::abcdef123456",
  "question": "string",
  "choices": {
    "A": "string",
    "B": "string",
    "C": "string",
    "D": "string"
  },
  "gold": "A|B|C|D"
}
```

QID list schema (`medqa_test_qids_seed42_n1000.json`):

```json
[
  "medqa::test::000123::abcdef123456",
  "..."
]
```

---

### 2) PubMedQA prepared subset

File:

- `phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl`

Purpose:

- prepared labeled PubMedQA subset for frozen generation and Phase-2 evaluation

Source note:

- the preparation script documents the upstream source as the `pqa_labeled` train split
- the released repository artifact is the prepared 1,000-example subset `pubmedqa_labeled_phase2.jsonl`
- where thesis-level wording refers more generically to frozen 1,000-example evaluation subsets, this should be understood as describing the released prepared artifact rather than renaming the upstream PubMedQA split

JSONL schema:

```json
{
  "qid": "string",
  "question": "string",
  "context": "string",
  "gold": "yes|no|maybe"
}
```

---

## Frozen Prediction Files (`outputs/frozen/`)

Typical files:

- `medqa_mistral7b.jsonl`
- `medqa_biomistral7b.jsonl`
- `pubmedqa_mistral7b.jsonl`
- `pubmedqa_biomistral7b.jsonl`

Purpose:

- one frozen prediction per example
- direct scoring input to `src/run_phase2.py`

Schema:

```json
{
  "qid": "string",
  "task": "medqa|pubmedqa",
  "question": "string",
  "context": "string",
  "choices": { "A": "...", "B": "...", "C": "...", "D": "..." },
  "gold": "string",
  "model_answer": "string",
  "pred": "string|null",
  "is_error": 0,
  "meta": {
    "model": "string",
    "max_new_tokens": 4,
    "temperature": 0.0,
    "do_sample": false,
    "top_p": 1.0,
    "prompt_truncation_max_length": 2048,
    "prompt_chars": 1234
  }
}
```

Notes:

- `pred` is derived by constrained parsing from generated continuation.
- rare constrained parsing failures may leave `pred` missing.
- in scoring, rare MedQA missing-`pred` cases can fall back to frozen `model_answer` text as teacher-forced answer sequence.
- `is_error` is the binary benchmark-defined target used in Phase-2 evaluation (`1 = incorrect`, `0 = correct`).
- `meta.prompt_truncation_max_length` records generation-side prompt truncation metadata from the frozen-output stage; scorer-side reruns may additionally apply their own context limits during feature extraction

---

## Final Result Files (`outputs/final/`)

Per run:

- `<run>.results.jsonl` (example-level scores)
- `<run>.manifest.json` (config, aggregate metrics, metadata)
- `<run>.manifest.bootstrap_indices.npz` (serialized bootstrap indices)

Example-level results schema:

```json
{
  "qid": "string",
  "task": "medqa|pubmedqa",
  "label": 0,
  "gold": "string|null",
  "pred": "string|null",
  "model_answer": "string",
  "lntp": 0.123,
  "mtp": 0.456,
  "egh_grad_norm": 0.0,
  "egh_emb_diff": 0.0,
  "egh_kl": 0.0,
  "egh_ce": 0.0,
  "egh_entropy": 0.0,
  "egh_probe_oof": 0.0,
  "egh_probe_ge": 0.0,
  "egh_probe_g_only": 0.0,
  "egh_probe_e_only": 0.0,
  "egh_probe_scalar_only": 0.0,
  "hidden_probe_oof": 0.0,
  "meta": {}
}
```

Manifest typically includes:

- run config (`task`, model, seed, splits, bootstrap budget, token-budget settings)
- aggregate metrics (AUROC, Spearman, CI, delta vs random)
- software/version metadata (Python, torch, transformers, sklearn, numpy)
- hidden-feature coverage metadata
- output paths

Manifest path note:

- absolute paths stored in manifests reflect the original execution environment and serve as archival metadata for auditability
- these path strings are not expected to resolve unchanged on every machine or checkout location

### Score Fields

The principal example-level score fields are:

- `lntp`: intrinsic logit-based score derived from answer-token probabilities using length normalization
- `mtp`: intrinsic logit-based score defined by the minimum realized answer-token probability
- `egh_grad_norm`, `egh_emb_diff`, `egh_kl`, `egh_ce`, `egh_entropy`: scalar gradient-, embedding-, or distribution-based primitives from the conditional/unconditional comparison used in the EGH-style feature family
- `egh_probe_oof`: out-of-fold supervised prediction from the main EGH-style feature-based probe
- `hidden_probe_oof`: out-of-fold supervised prediction from the pooled hidden-state probe
- `egh_probe_ge`, `egh_probe_g_only`, `egh_probe_e_only`, `egh_probe_scalar_only`: released comparison variants of the EGH probe family used for ablation and analysis rather than separate benchmark targets

### Result Families

Within `outputs/figures_tables/`:

- `tables_general/` and `figures_general/` contain baseline summary exports derived from `outputs/final/`
- `ablations/` mirrors the major ablation families and stores derived summaries for those runs

---

## Label Semantics

Primary binary target:

- `is_error = 1`: frozen prediction is incorrect vs benchmark gold
- `is_error = 0`: frozen prediction is correct vs benchmark gold

In scorer outputs this is mirrored as `label`.

This target is a constrained benchmark prediction-error label, not a direct response-level hallucination label in the Phase-1 sense.

---

## Determinism and Reproducibility Notes

- Phase-2 evaluation uses frozen predictions (no regeneration during scoring).
- bootstrap index arrays are serialized for reproducible CI regeneration from archived score artifacts.
- out-of-fold probe scoring uses fixed seeds and stratified folds.
- small floating-point differences can still occur across runtime environments; reported outputs are tied to archived artifacts.

---

## Evaluation Guardrails (High Level)

The documented validity controls for this evaluation setting include:

- finiteness checks on scorer outputs
- rejection of degenerate constant score vectors where ranking metrics would not be informative
- explicit tracking of invalid or excluded examples
- preservation of raw score outputs prior to polarity alignment where applicable

These controls support auditability, but the archived `results.jsonl` and `manifest.json` files remain the authoritative interpretation artifacts.

---

## Data Validation Checklist (Recommended)

Before running analysis:

1. verify required `outputs/frozen/*.jsonl` files exist
2. verify row counts (typically 1000 per task/model frozen file)
3. verify `qid` uniqueness within each file
4. verify `is_error` exists and is binary
5. verify each final run has matching `results + manifest + bootstrap_indices`

---

## Related Docs

- Reproduction guide: `phase_2_medical/reproduce_phase2.md`
- Project overview: `phase_2_medical/README.md`
