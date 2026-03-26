# Token Score Bias Ablation

## Overview

This directory contains a methodological ablation examining potential length-related bias in the token-level uncertainty scores LNTP and MTP.

The ablation asks whether score magnitude is influenced by answer length or by the amount of answer text included in scoring. Its purpose is to test whether these token-level scores reflect uncertainty and hallucination-related signal, rather than primarily capturing superficial length effects.

---

## Hypothesis

Unnormalized token-level scores based on summed token contributions may show stronger dependence on answer length than length-normalized mean scores. If this dependence is substantial, some apparent detection performance may reflect answer-length sensitivity rather than uncertainty alone.

A complementary answer-span sweep tests whether LNTP and MTP are sensitive to how much of the answer text is scored.

---

## Motivation / Background

LNTP and MTP are token-level white-box scores, so their aggregation rule matters conceptually. A sum-based score can increase simply because more answer tokens are present, whereas a mean-based score is intended to reduce this dependence by normalizing for length.

This distinction is important for uncertainty quantification and hallucination detection. If token-level score magnitude is strongly driven by answer length, then part of the measured signal may be attributable to a structural property of the response rather than to genuine uncertainty. Similarly, if performance changes substantially when only an answer prefix is scored, then interpretation of the score may depend on answer-span coverage.

---

## Ablation Design

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

The archived runs in this directory use frozen TruthfulQA hallucination annotations converted to the Phase 2 frozen-input format. The current released artifacts in this folder cover the Mistral configuration.

Operationally, the ablation is implemented with:

- `phase_2_medical/src/run_token_bias_lntp_mtp.py`
- `phase_2_medical/scripts/run_ablation_token_score_bias.sh`

The maintained runner keeps the model and runtime configuration fixed while computing the mean-versus-sum comparison and the answer-prefix sweep. Each model directory contains per-example score outputs and summary manifests for direct inspection.

---

## Expected Effect

If token-length bias is present, sum-based variants should exhibit stronger dependence on answer length than mean-based variants. In that case, any apparent detection advantage associated with unnormalized sums should be interpreted cautiously.

If the answer-span sweep shows marked changes as `k` varies, this would indicate that LNTP or MTP depends materially on how much answer content is included in scoring. For uncertainty quantification and hallucination detection, this matters because strong length or span sensitivity would complicate interpretation of these scores as content-sensitive uncertainty measures.

---

## Notes / Interpretation

This ablation should be interpreted as a methodological appendix analysis of score construction and score validity. It is not a primary comparison within the main MedQA and PubMedQA evaluation grid.

Accordingly, the main question is whether LNTP and MTP remain interpretable after accounting for answer length and answer-span effects. If strong dependence on length or prefix coverage is observed, then conclusions based on these token-level scores should be qualified accordingly.