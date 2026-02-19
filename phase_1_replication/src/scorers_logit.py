# scorers_logit.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import torch

from modeling_llm import LLMWrapper, BatchEncodedQA


Orientation = Literal["confidence", "uncertainty"]


def encode_qa_batch(
    llm: LLMWrapper,
    questions: Sequence[str],
    answers: Sequence[str],
    *,
    cache: Optional[Dict[Tuple[str, str], BatchEncodedQA]] = None,
) -> BatchEncodedQA:
    """
    Must-Fix 1 + 2:
    Delegate to llm.encode_qa_batch (single source of truth).
    Keep optional cache behavior intact for repeated (q, a) pairs.
    """
    if len(questions) != len(answers):
        raise ValueError(
            f"questions and answers must have same length, got {len(questions)} vs {len(answers)}"
        )
    B = len(questions)
    if B == 0:
        raise ValueError("encode_qa_batch received empty batch.")
    
    # Optional cache for repeated (q, a) pairs
    if cache is not None:
        cached = []
        for q, a in zip(questions, answers):
            item = cache.get((q, a))
            if item is None:
                break
            cached.append(item)

        if len(cached) == B:
            # Concatenate along batch dimension for all tensors in the contract
            return BatchEncodedQA(
                input_ids=torch.cat([c.input_ids for c in cached], dim=0),
                attention_mask=torch.cat([c.attention_mask for c in cached], dim=0),
                answer_start_idx=torch.cat([c.answer_start_idx for c in cached], dim=0),
                prompt_len=torch.cat([c.prompt_len for c in cached], dim=0),
                seq_len=torch.cat([c.seq_len for c in cached], dim=0),
                first_real_idx=torch.cat([c.first_real_idx for c in cached], dim=0),
                answer_len=torch.cat([c.answer_len for c in cached], dim=0),
            )

    out = llm.encode_qa_batch(questions, answers)

    if cache is not None:
        for i, (q, a) in enumerate(zip(questions, answers)):
            cache[(q, a)] = BatchEncodedQA(
                input_ids=out.input_ids[i:i+1].detach(),
                attention_mask=out.attention_mask[i:i+1].detach(),
                answer_start_idx=out.answer_start_idx[i:i+1].detach(),
                prompt_len=out.prompt_len[i:i+1].detach(),
                seq_len=out.seq_len[i:i+1].detach(),
                first_real_idx=out.first_real_idx[i:i+1].detach(),
                answer_len=out.answer_len[i:i+1].detach(),
            )

    return out


# =============================================================================
# Core: token log-probs for ANSWER tokens only (supports BATCH)
# =============================================================================

def _get_answer_token_logprobs_batch(
    *,
    logits: torch.Tensor,
    enc: BatchEncodedQA,
) -> List[torch.Tensor]:
    """
    Extract log-probabilities of the realized ANSWER tokens under teacher forcing.

    Shapes:
      logits:          (B, T, V)
      enc.input_ids:   (B, T)
      enc.attention_mask: (B, T)  (1=real token, 0=pad)
      enc.answer_start_idx: (B,) token index in PADDED space where answer begins

    Autoregressive alignment:
      Token at position t is predicted by logits at position (t-1).
    """
    input_ids = enc.input_ids
    answer_start_idx = enc.answer_start_idx
    first_real_idx = enc.first_real_idx
    seq_len = enc.seq_len

    B, T, V = logits.shape

    # Shift so logits at t predict token at t+1
    shifted_logits = logits[:, :-1, :]         # (B, T-1, V)
    target_ids = input_ids[:, 1:]              # (B, T-1)
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    realized_logp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    out: List[torch.Tensor] = []
    for i in range(B):
        # realized_logp is shifted by 1, hence "- 1"
        start = int(answer_start_idx[i].item()) - 1

        # end_real is exclusive end index of real tokens in PADDED space
        end_real = int((first_real_idx[i] + seq_len[i]).item())
        end = end_real - 1

        if start < 0 or start >= end:
            out.append(torch.empty(0, device=logits.device))
        else:
            out.append(realized_logp[i, start:end])

    return out


def lntp_from_logp(logps: torch.Tensor) -> float:
    if logps.numel() == 0:
        return float("nan")
    return float(torch.exp(logps.mean()).item())


def mtp_from_logp(logps: torch.Tensor) -> float:
    if logps.numel() == 0:
        return float("nan")
    return float(torch.exp(logps.min()).item())


def logprob_stats_from_logp(logps: torch.Tensor) -> Dict[str, float]:
    if logps.numel() == 0:
        return {"n_answer_tokens": 0, "mean_logp": 0.0, "min_logp": 0.0}
    return {
        "n_answer_tokens": int(logps.numel()),
        "mean_logp": float(logps.mean().item()),
        "min_logp": float(logps.min().item()),
    }


@torch.no_grad()
def compute_lntp_mtp_for_qa_batch(
    llm: LLMWrapper,
    questions: Sequence[str],
    answers: Sequence[str],
    *,
    orientation: str = "uncertainty",
    return_log_stats: bool = False,
    cache: Optional[Dict[Tuple[str, str], BatchEncodedQA]] = None,
) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
    if orientation not in ("uncertainty", "confidence"):
        raise ValueError("orientation must be 'uncertainty' or 'confidence'")

    batch = encode_qa_batch(llm, questions, answers, cache=cache)

    out = llm.forward(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        output_hidden_states=False,
    )

    logps_list = _get_answer_token_logprobs_batch(
        logits=out.logits,
        enc=batch,
    )

    lntp_vals: List[float] = []
    mtp_vals: List[float] = []
    stats_vals: List[Dict[str, float]] = []

    # Diagnostics: count empty/invalid answer spans (must be zero for paper-ready runs)
    n_empty = 0
    min_len: Optional[int] = None
    max_len: Optional[int] = None

    for logps in logps_list:
        if logps.numel() == 0:
            n_empty += 1
        else:
            n = int(logps.numel())
            min_len = n if (min_len is None) else min(min_len, n)
            max_len = n if (max_len is None) else max(max_len, n)

        l = lntp_from_logp(logps)
        m = mtp_from_logp(logps)

        # Convert to uncertainty if requested
        if orientation == "uncertainty":
            l = 1.0 - l
            m = 1.0 - m

        lntp_vals.append(float(l))
        mtp_vals.append(float(m))

        if return_log_stats:
            stats_vals.append(logprob_stats_from_logp(logps))

    # Paper-ready guardrail: never allow silent empty spans in batch scoring.
    if n_empty > 0:
        raise RuntimeError(
            f"LNTP/MTP: detected {n_empty}/{len(logps_list)} empty/invalid answer spans in batch scoring. "
            f"Non-empty answer token lengths range: [{min_len}, {max_len}]. "
            "Fix prompt/answer alignment or answer_start_idx calculation; do not treat this as score=0."
        )

    if return_log_stats:
        return lntp_vals, mtp_vals, stats_vals

    return lntp_vals, mtp_vals, []