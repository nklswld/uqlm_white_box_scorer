# src/scorers_gradient.py
"""
Gradient-based (EGH-inspired) primitives for hallucination detection (Hu et al., 2024).

Paper-ready goals of this rewrite:
- Alignment correctness: uses LLMWrapper.encode_qa() as the single source of truth.
- Hu-style features:
    * d_loss    = mean token-wise KL( P(.|Q) || P(.|0) ) over answer prediction positions
    * ce_loss   = mean token-wise cross-entropy H( P(.|Q), P(.|0) ) over answer prediction positions
    * h_p       = mean token-wise entropy H( P(.|Q) ) over answer prediction positions
    * emb_diff  = mean L2 distance between LAST-LAYER hidden states for answer tokens (E feature)
    * grad_norm = mean L2 norm of gradients wrt UNCONDITIONAL input embeddings for answer tokens,
                  where the loss is KL( P(.|Q) || P(.|0) ) (G feature)
- Avoid silent failures:
    * strict shape checks + deterministic fallbacks if strict=False
- Batching/Caching hooks:
    * compute_egh_primitives_batch() computes d_loss + ce_loss + h_p + emb_diff in batches (fast path)
      (grad_norm remains per-sample by default due to memory/graph complexity)

IMPORTANT:
- This file returns RAW primitives. For the Phase-1 runner, your “hallucination-likely” direction
  should be handled at the scorer/feature-mapping layer (e.g., by negating KL/CE if desired).

"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from modeling_llm import LLMWrapper

# ---------------------------------------------------------------------
# Score orientation contract (central, explicit)
# ---------------------------------------------------------------------

ORIENTATION_CONTRACT: Dict[str, str] = {
    # Hu-style divergences: often hallu -> LOWER divergence/CE (closer to uncond).
    # Keep raw primitives here; direction can be enforced downstream if desired.
    "d_loss": "raw",
    "ce_loss": "raw",
    "h_p": "raw",
    "emb_diff": "raw",
    "grad_norm": "raw",
}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _safe_pad_token_id(llm: LLMWrapper) -> int:
    tok = llm.tokenizer
    if getattr(tok, "pad_token_id", None) is not None:
        return int(tok.pad_token_id)
    if getattr(tok, "eos_token_id", None) is not None:
        return int(tok.eos_token_id)
    return 0


def _encode_cond_uncond(
    llm: LLMWrapper,
    question: str,
    answer: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int, int, torch.Tensor]:
    """
    Build aligned conditional/unconditional inputs using the encode_qa alignment contract.

    conditional:   [prompt(question) + answer] with attention_mask=1
    unconditional: [PAD...PAD + answer] with attention_mask=1
      - We intentionally keep attention_mask=1 on the prefix to avoid a silent alignment bug where
        the logits predicting the first answer token would otherwise come from a masked position.

    Returns:
      inputs_cond, inputs_uncond, ans_start, a_len, answer_token_ids
    """
    device = llm.input_device
    tok = llm.tokenizer
    pad_id = _safe_pad_token_id(llm)

    # Single source of truth for prompt and answer start index.
    enc = llm.encode_qa(question, answer)
    input_ids_c = enc["input_ids"]  # (1, seq_len)
    attn_c = enc["attention_mask"]  # (1, seq_len)
    ans_start = int(enc["answer_start_idx"])

    seq_len = int(input_ids_c.size(1))
    if ans_start < 0 or ans_start > seq_len:
        raise ValueError(f"answer_start_idx out of range: {ans_start} for seq_len={seq_len}")

    # FIX (minimal): derive answer token ids directly from conditional suffix to guarantee token-boundary consistency.
    # Answer tokens occupy positions [ans_start : seq_len) in input_ids_c.
    a_ids_t = input_ids_c[:, ans_start:seq_len]
    a_len = int(a_ids_t.size(1))

    if a_len == 0:
        dummy = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        inputs = {"input_ids": dummy, "attention_mask": torch.ones_like(dummy)}
        return (
            inputs,
            inputs,
            ans_start,
            0,
            torch.zeros((1, 0), dtype=torch.long, device=device),
        )

    # Unconditional: prefix length must match ans_start for perfect alignment.
    prefix = torch.full((1, ans_start), pad_id, dtype=torch.long, device=device)
    input_ids_u = torch.cat([prefix, a_ids_t.to(device)], dim=1)  # (1, ans_start + a_len)

    # Keep attention_mask=1 everywhere to ensure "predict first answer token" position is valid.
    attn_u = torch.ones_like(input_ids_u, dtype=torch.long, device=device)

    inputs_cond = {"input_ids": input_ids_c, "attention_mask": attn_c}
    inputs_uncond = {"input_ids": input_ids_u, "attention_mask": attn_u}
    answer_token_ids = a_ids_t.to(device)  # (1, a_len)

    return inputs_cond, inputs_uncond, ans_start, a_len, answer_token_ids


def _prediction_slice(ans_start: int, a_len: int) -> Tuple[int, int]:
    """
    For a causal LM, logits at position t predict token at position t+1.
    Answer tokens occupy positions [ans_start, ans_start + a_len - 1].
    So the logits predicting those answer tokens live at positions:
      pred_start = ans_start - 1
      pred_end   = ans_start + a_len - 1   (exclusive end index)
    """
    pred_start = ans_start - 1
    pred_end = ans_start + a_len - 1
    return pred_start, pred_end


def _kl_tokenwise_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise KL(P || Q) for distributions from logits.
    Inputs: (B, T, V)
    Output: (B, T)
    """
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return torch.sum(p * (logp - logq), dim=-1)


def _cross_entropy_tokenwise_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise cross-entropy H(P, Q) where P comes from logits_p and Q from logits_q.
    Inputs: (B, T, V)
    Output: (B, T)
    H(P,Q) = - sum_x P(x) log Q(x)
    """
    logq = torch.log_softmax(logits_q, dim=-1)
    p = torch.softmax(logits_p, dim=-1)
    return -torch.sum(p * logq, dim=-1)


def _entropy_tokenwise_from_logits(logits_p: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise entropy H(P) where P comes from logits_p.
    Inputs: (B, T, V)
    Output: (B, T)
    H(P) = - sum_x P(x) log P(x)
    """
    logp = torch.log_softmax(logits_p, dim=-1)
    p = logp.exp()
    return -torch.sum(p * logp, dim=-1)


def _common_answer_len(
    seq_len_c: int,
    seq_len_u: int,
    ans_start: int,
    a_len: int,
) -> int:
    """
    Return the maximal common answer-token length L such that both
    conditional and unconditional sequences contain [ans_start : ans_start + L).
    """
    if a_len <= 0:
        return 0
    if ans_start < 0:
        return 0
    avail_c = seq_len_c - ans_start
    avail_u = seq_len_u - ans_start
    L = min(a_len, avail_c, avail_u)
    return int(max(0, L))


# ---------------------------------------------------------------------
# Batched forward primitives (fast path)
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_egh_primitives_batch(
    llm: LLMWrapper,
    qa_pairs: Sequence[Tuple[str, str]],
    *,
    batch_size: int = 8,
    strict: bool = True,
) -> List[Dict[str, float]]:
    """
    Batched computation of:
      - d_loss (KL gap)
      - ce_loss (cross-entropy H(P(.|Q), P(.|0)))
      - h_p (entropy H(P(.|Q)))
      - emb_diff (last-layer hidden state diff on answer tokens)

    grad_norm is NOT computed here (set to 0.0), because it requires graphs/backward.
    Use compute_egh_primitives_for_qa(...) for grad_norm per sample.

    Returns list aligned with qa_pairs.
    """
    model = llm.model
    model.eval()

    results: List[Dict[str, float]] = []

    try:
        for s in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[s : s + batch_size]

            cond_feats = []
            uncond_feats = []
            meta: List[Tuple[int, int]] = []  # (ans_start, a_len)

            for q, a in batch:
                inputs_c, inputs_u, ans_start, a_len, _ = _encode_cond_uncond(llm, q, a)
                cond_feats.append(
                    {
                        "input_ids": inputs_c["input_ids"].squeeze(0),
                        "attention_mask": inputs_c["attention_mask"].squeeze(0),
                    }
                )
                uncond_feats.append(
                    {
                        "input_ids": inputs_u["input_ids"].squeeze(0),
                        "attention_mask": inputs_u["attention_mask"].squeeze(0),
                    }
                )
                meta.append((ans_start, a_len))

            tok = llm.tokenizer
            cond_batch = tok.pad(cond_feats, return_tensors="pt").to(llm.input_device)
            uncond_batch = tok.pad(uncond_feats, return_tensors="pt").to(llm.input_device)

            out_c = model(**cond_batch, output_hidden_states=True, use_cache=False)
            out_u = model(**uncond_batch, output_hidden_states=True, use_cache=False)

            logits_c = out_c.logits
            logits_u = out_u.logits
            h_c_last = out_c.hidden_states[-1]
            h_u_last = out_u.hidden_states[-1]

            # Sanity: batch dims match
            if logits_c.size(0) != logits_u.size(0) or h_c_last.size(0) != h_u_last.size(0):
                if strict:
                    raise RuntimeError("Batch size mismatch between conditional/unconditional forward outputs.")
                # fallback: zeros for this batch
                results.extend(
                    [
                        {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []}
                        for _ in batch
                    ]
                )
                continue

            for i, (ans_start, a_len) in enumerate(meta):
                if a_len <= 0:
                    results.append({"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []})
                    continue

                pred_start, pred_end = _prediction_slice(ans_start, a_len)
                if pred_start < 0 or pred_end <= pred_start:
                    results.append({"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []})
                    continue

                # Slice logits once
                lc = logits_c[i : i + 1, pred_start:pred_end, :]
                lu = logits_u[i : i + 1, pred_start:pred_end, :]

                # d_loss: KL gap on answer prediction positions
                kl_tok = _kl_tokenwise_from_logits(lc, lu)
                d_loss = float(kl_tok.mean().item())

                # ce_loss: cross-entropy H(P(.|Q), P(.|0))
                ce_tok = _cross_entropy_tokenwise_from_logits(lc, lu)
                ce_loss = float(ce_tok.mean().item())

                # h_p: entropy H(P(.|Q))
                h_tok = _entropy_tokenwise_from_logits(lc)
                h_p = float(h_tok.mean().item())

                # emb_diff: robustly compare only the maximal common answer span
                seq_len_c = int(h_c_last.size(1))
                seq_len_u = int(h_u_last.size(1))
                L = _common_answer_len(seq_len_c, seq_len_u, ans_start, a_len)

                if L <= 0:
                    emb_diff = 0.0
                else:
                    h_c_ans = h_c_last[i : i + 1, ans_start : ans_start + L, :]
                    h_u_ans = h_u_last[i : i + 1, ans_start : ans_start + L, :]
                    emb_diff = float(torch.norm(h_c_ans - h_u_ans, dim=-1).mean().item())

                results.append(
                    {"d_loss": d_loss, "ce_loss": ce_loss, "h_p": h_p, "emb_diff": emb_diff, "grad_norm": 0.0, "g_vec": [], "e_vec": []}
                )

        return results

    except Exception:
        if strict:
            raise
        return [{"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []} for _ in qa_pairs]


# ---------------------------------------------------------------------
# Main primitive (single sample; includes gradient feature)
# ---------------------------------------------------------------------


def compute_egh_primitives_for_qa(
    llm: LLMWrapper,
    question: str,
    answer: str,
    *,
    strict: bool = True,
    chunk_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute EGH-inspired primitives (feature-level signals), NOT the full EGH detector:

      - d_loss    : mean token-wise KL( P(.|Q) || P(.|0) ) over answer prediction positions
      - ce_loss   : mean token-wise cross-entropy H( P(.|Q), P(.|0) ) over answer prediction positions
      - h_p       : mean token-wise entropy H( P(.|Q) ) over answer prediction positions
      - emb_diff  : mean L2 distance between conditional vs unconditional LAST-LAYER hidden states
                   (answer token positions)  -> Hu-style "E"
      - grad_norm : mean L2 norm of gradients wrt UNCONDITIONAL input embeddings (answer positions),
                   where the loss is the KL above -> Hu-style "G"

    chunk_size:
      - None (default): single backward over full answer span
      - int: chunked loss accumulation with retain_graph=True (only if needed)

    Output contract:
      Returns RAW floats. Any direction/orientation used for “hallucination-likely”
      should be applied downstream (e.g., notebook or feature mapping).
    """
    try:
        model = llm.model
        model.eval()

        inputs_c, inputs_u, ans_start, a_len, _ = _encode_cond_uncond(llm, question, answer)
        if a_len <= 0:
            return {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "grad_norm": 0.0, "emb_diff": 0.0, "g_vec": [], "e_vec": []}
        
        pred_start, pred_end = _prediction_slice(ans_start, a_len)
        if pred_start < 0 or pred_end <= pred_start:
            return {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "grad_norm": 0.0, "emb_diff": 0.0, "g_vec": [], "e_vec": []}

        # ------------------------------------------------------------------
        # (1) d_loss + (ce_loss) + (h_p) + (2) emb_diff (no grad)
        # ------------------------------------------------------------------
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True, use_cache=False)
            out_u = model(**inputs_u, output_hidden_states=True, use_cache=False)

            logits_c = out_c.logits[:, pred_start:pred_end, :]
            logits_u = out_u.logits[:, pred_start:pred_end, :]

            kl_tok = _kl_tokenwise_from_logits(logits_c, logits_u)  # (1, T_pred)
            d_loss = float(kl_tok.mean().item())

            ce_tok = _cross_entropy_tokenwise_from_logits(logits_c, logits_u)  # (1, T_pred)
            ce_loss = float(ce_tok.mean().item())

            h_tok = _entropy_tokenwise_from_logits(logits_c)  # (1, T_pred)
            h_p = float(h_tok.mean().item())

            # Hu-style E: last hidden layer as "embedding"
            h_c_last = out_c.hidden_states[-1]  # (1, seq_len_c, H)
            h_u_last = out_u.hidden_states[-1]  # (1, seq_len_u, H)

            seq_len_c = int(h_c_last.size(1))
            seq_len_u = int(h_u_last.size(1))
            L = _common_answer_len(seq_len_c, seq_len_u, ans_start, a_len)



            if L <= 0:
                emb_diff = 0.0
                e_vec = torch.zeros((0,), device=llm.input_device)
            else:
                h_c_ans = h_c_last[:, ans_start : ans_start + L, :]
                h_u_ans = h_u_last[:, ans_start : ans_start + L, :]

                # Scalar diagnostic
                emb_diff = float(torch.norm(h_c_ans - h_u_ans, dim=-1).mean().item())

                # E_vector (fixed-size): mean over answer tokens -> (H,)
                e_vec = (h_c_ans - h_u_ans).mean(dim=1).squeeze(0)   # shape (H,)


            # Detach conditional logits for gradient loss
            logits_c_det = logits_c.detach()

        # ------------------------------------------------------------------
        # (3) grad_norm (KL-gradient wrt unconditional input embeddings)
        # ------------------------------------------------------------------
        input_embeddings = model.get_input_embeddings()

        # Build grad-enabled unconditional embeddings
        emb_u = input_embeddings(inputs_u["input_ids"]).detach()
        emb_u.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        out_u_emb = model(
            inputs_embeds=emb_u,
            attention_mask=inputs_u.get("attention_mask", None),
            use_cache=False,
        )

        logits_u_emb = out_u_emb.logits[:, pred_start:pred_end, :]  # (1, T_pred_u, V)

        # Ensure time alignment for KL
        if logits_u_emb.size(1) != logits_c_det.size(1):
            if strict:
                raise RuntimeError(
                    f"Alignment mismatch: logits_u_emb T={logits_u_emb.size(1)} != logits_c_det T={logits_c_det.size(1)}"
                )
            return {
                "d_loss": float(d_loss),
                "ce_loss": float(ce_loss),
                "h_p": float(h_p),
                "grad_norm": 0.0,
                "emb_diff": float(emb_diff),
                "g_vec": [],
                "e_vec": e_vec.detach().to(torch.float32).cpu().tolist() if "e_vec" in locals() else [],
            }

        logp_c = torch.log_softmax(logits_c_det, dim=-1)
        p_c = logp_c.exp()
        logq_u = torch.log_softmax(logits_u_emb, dim=-1)
        kl_per_tok = torch.sum(p_c * (logp_c - logq_u), dim=-1)  # (1, T_pred)

        if chunk_size is None:
            loss = kl_per_tok.mean()
            loss.backward()
        else:
            cs = int(chunk_size)
            T_pred = int(kl_per_tok.size(1))
            for s in range(0, T_pred, cs):
                e = min(T_pred, s + cs)
                loss_chunk = kl_per_tok[:, s:e].mean()
                loss_chunk.backward(retain_graph=True)

        if emb_u.grad is None:
            raise RuntimeError("Gradient is None after backward() (unexpected).")

        # grad over maximal common answer span (same principle as emb_diff)
        seq_len_u_emb = int(emb_u.size(1))
        # For grad, only unconditional length matters (it’s the only one with grads),
        # but we still clamp by a_len and available tokens.
        Lg = _common_answer_len(seq_len_u_emb, seq_len_u_emb, ans_start, a_len)

        if Lg <= 0:
            grad_norm = 0.0
            g_vec = torch.zeros((0,), device=llm.input_device)
        else:
            grad_ans = emb_u.grad[:, ans_start : ans_start + Lg, :]  # (1, Lg, H)

            # Scalar diagnostic
            grad_norm = float(grad_ans.norm(dim=-1).mean().item())

            # G_vector (fixed-size): mean over answer tokens -> (H,)
            g_vec = grad_ans.mean(dim=1).squeeze(0)  # shape (H,)

        return {
            "d_loss": float(d_loss),
            "ce_loss": float(ce_loss),
            "h_p": float(h_p),
            "grad_norm": float(grad_norm),
            "emb_diff": float(emb_diff),
            "g_vec": g_vec.detach().to(torch.float32).cpu().tolist(),
            "e_vec": e_vec.detach().to(torch.float32).cpu().tolist(),
        }

    except Exception:
        if strict:
            raise  
        return {
            "d_loss": 0.0,
            "ce_loss": 0.0,
            "h_p": 0.0,
            "grad_norm": 0.0,
            "emb_diff": 0.0,
            "g_vec": [],
            "e_vec": [],
        }
