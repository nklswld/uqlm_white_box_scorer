# Token Score Bias Ablation

## Aim

This ablation examines whether the token-level scores LNTP and MTP are sensitive to answer length or answer-span coverage. The concern is methodological validity: a score that changes mainly because more answer tokens are included is harder to interpret as a content-sensitive signal.

---

## Analyses

This ablation contains two complementary analyses.

### 1. Length-normalization analysis

LNTP and MTP are evaluated in both mean and sum form:

- **LNTP_mean**
- **LNTP_sum**
- **MTP_mean**
- **MTP_sum**

For these variants, the analysis records:

- AUROC for hallucination detection
- Spearman correlation between score magnitude and answer length

### 2. Answer-span sensitivity analysis

The analysis also performs a prefix-length sweep by scoring only the first `k` answer tokens for:

- **k = 1, 2, 3, 5, 10, 20**

This tests how sensitive token-level detection performance is to the amount of answer content included.

The archived runs in this directory use frozen TruthfulQA hallucination annotations converted to the Phase 2 frozen-input format. The released artifacts in this folder cover the Mistral configuration.

Implementation entry points:

- `phase_2_medical/src/run_token_bias_lntp_mtp.py`
- `phase_2_medical/scripts/run_ablation_token_score_bias.sh`

The maintained runner keeps the model and runtime configuration fixed while computing both the mean-versus-sum comparison and the answer-prefix sweep. Each model directory contains per-example score outputs and summary manifests.

---

## Why It Matters

If token-length bias is present, sum-based variants should show stronger dependence on answer length than mean-based variants. In that case, any apparent detection advantage associated with unnormalized sums needs qualification.

If the answer-span sweep changes markedly with `k`, LNTP or MTP depends materially on how much of the answer is scored. That would complicate interpretation of these scores as content-sensitive signals.

---

## Reading the Results

This directory is best read as a methodological appendix analysis of score construction. It is not part of the main MedQA/PubMedQA comparison grid. The main question is whether LNTP and MTP remain interpretable once answer length and answer-span effects are made explicit.
