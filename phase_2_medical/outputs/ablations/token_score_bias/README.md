# Token Score Bias Ablation

This directory contains the ablation study investigating  
potential token-length bias in token-level uncertainty scores  
(LNTP and MTP).

---

## Hypothesis

Token-level uncertainty scores may correlate with answer length.

In particular:

- **Unnormalized scores (sum)** may systematically increase with
  answer length.
- **Normalized scores (mean)** should reduce or eliminate this bias.

If a strong correlation exists between score magnitude and
token length, the probe may partially capture length effects
instead of genuine uncertainty.

---

## Ablation Design

Two complementary analyses are performed:

1. **Mean vs. Sum Comparison**
   - LNTP_mean vs LNTP_sum
   - MTP_mean vs MTP_sum  
   Evaluated via AUROC.

2. **Length Correlation Analysis**
   - Spearman correlation between score and answer length
   - Computed separately for:
     - LNTP_mean
     - LNTP_sum
     - MTP_mean
     - MTP_sum

The dataset used is:

- TruthfulQA hallucination annotations  
  (converted to Phase 2 frozen format)

All model parameters and seeds are held constant.

---

## Implementation

The ablation is implemented using:

- `run_token_bias_lntp_mtp.py`
- `run_ablation_token_score_bias.sh`

Relevant arguments:

- `--frozen_jsonl <path>`
- `--model_name <HF model>`
- `--seed 42`
- `--device cuda:0`
- `--dtype bfloat16`

The script computes:

- LNTP (mean and sum)
- MTP (mean and sum)
- Spearman correlations with answer length
- AUROC for hallucination detection

---

## Outputs

Each model directory contains:

### `results.jsonl`

Per-example outputs including:

- answer length (tokens)
- LNTP_mean
- LNTP_sum
- MTP_mean
- MTP_sum

### `manifest.json`

Summary statistics including:

- Spearman correlations
- AUROC scores
- dataset statistics
- environment versions

---

## Interpretation Goal

If:

- **sum-based scores** show strong positive correlation with
  answer length, and
- **mean-based scores** reduce this effect,

then token-length normalization is justified.

If correlations are weak and AUROC differences are negligible,
length bias is unlikely to meaningfully affect results.