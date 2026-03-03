"""
Compute gradient- and embedding-based primitives for QA hallucination scoring.
Inputs: (question:str, answer:str) pairs and an LLMWrapper providing encode_qa() alignment.
Outputs: per-sample raw floats (d_loss, ce_loss, h_p, emb_diff, grad_norm) plus optional fixed-size vectors (e_vec, g_vec).
The primitives are "orientation-agnostic": no sign/polarity is applied here; downstream scorers map to "hallucination-likely".
Determinism: forward-only paths are deterministic given model + inputs; grad_norm depends on exact graph/precision and must be
recomputed per sample (no randomness introduced here).
"""

# phase_1_replication/src/scorers_gradient.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from modeling_llm import LLMWrapper

# ---------------------------------------------------------------------
# Score orientation contract (central, explicit)
# ---------------------------------------------------------------------

# Convention: this module returns raw (unsigned) primitives; any "higher=more hallu" convention is enforced downstream.
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
    # Prefer tokenizer-defined pad; fall back to eos; final fallback is 0 (may be semantically wrong for some tokenizers).
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
    Build aligned conditional/unconditional inputs using llm.encode_qa() as source of truth.
    Returns: (inputs_cond, inputs_uncond, ans_start, a_len, answer_token_ids).
    """
    device = llm.input_device
    tok = llm.tokenizer
    pad_id = _safe_pad_token_id(llm)

    # Alignment contract: encode_qa defines prompt construction and answer_start_idx.
    enc = llm.encode_qa(question, answer)
    input_ids_c = enc["input_ids"]  # (1, seq_len)
    attn_c = enc["attention_mask"]  # (1, seq_len)
    ans_start = int(enc["answer_start_idx"])

    seq_len = int(input_ids_c.size(1))
    if ans_start < 0 or ans_start > seq_len:
        raise ValueError(f"answer_start_idx out of range: {ans_start} for seq_len={seq_len}")

    # Answer token ids are derived from the conditional suffix to preserve tokenizer boundary consistency.
    a_ids_t = input_ids_c[:, ans_start:seq_len]
    a_len = int(a_ids_t.size(1))

    if a_len == 0:
        # Degenerate case: no answer tokens -> deterministic zero primitives (avoid downstream shape assumptions).
        dummy = torch.tensor([[pad_id]], dtype=torch.long, device=device)
        inputs = {"input_ids": dummy, "attention_mask": torch.ones_like(dummy)}
        return (
            inputs,
            inputs,
            ans_start,
            0,
            torch.zeros((1, 0), dtype=torch.long, device=device),
        )

    # Unconditional input: same answer suffix, prefix padded to exactly ans_start for time-step alignment.
    prefix = torch.full((1, ans_start), pad_id, dtype=torch.long, device=device)
    input_ids_u = torch.cat([prefix, a_ids_t.to(device)], dim=1)  # (1, ans_start + a_len)

    # NOTE: potential issue: attention_mask=1 on padded prefix is intentional to avoid masking out the "predict first answer token" position.
    attn_u = torch.ones_like(input_ids_u, dtype=torch.long, device=device)

    inputs_cond = {"input_ids": input_ids_c, "attention_mask": attn_c}
    inputs_uncond = {"input_ids": input_ids_u, "attention_mask": attn_u}
    answer_token_ids = a_ids_t.to(device)  # (1, a_len)

    return inputs_cond, inputs_uncond, ans_start, a_len, answer_token_ids


def _prediction_slice(ans_start: int, a_len: int) -> Tuple[int, int]:
    """
    Map answer token positions to the causal-LM logit slice that predicts them (t predicts t+1).
    """
    # Invariant: logits[:, pred_start:pred_end] align to answer tokens [ans_start : ans_start+a_len).
    pred_start = ans_start - 1
    pred_end = ans_start + a_len - 1
    return pred_start, pred_end


def _kl_tokenwise_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise KL(P || Q) from logits, with P defined by logits_p and Q by logits_q.
    """
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return torch.sum(p * (logp - logq), dim=-1)


def _cross_entropy_tokenwise_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise cross-entropy H(P, Q) where P is from logits_p and Q is from logits_q.
    """
    logq = torch.log_softmax(logits_q, dim=-1)
    p = torch.softmax(logits_p, dim=-1)
    return -torch.sum(p * logq, dim=-1)


def _entropy_tokenwise_from_logits(logits_p: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise entropy H(P) where P is from logits_p.
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
    Compute the largest L such that both sequences contain answer span [ans_start : ans_start+L).
    """
    # Silent mismatch guard: unconditional may be shorter if upstream tokenization/alignment assumptions break.
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
    Batched forward-only primitives: d_loss, ce_loss, h_p, emb_diff (grad_norm is set to 0.0).
    Returns a list aligned 1:1 with qa_pairs; strict=False yields deterministic zero-filled fallbacks on failure.
    """
    model = llm.model
    model.eval()

    results: List[Dict[str, float]] = []

    try:
        for s in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[s : s + batch_size]

            cond_feats = []
            uncond_feats = []
            meta: List[Tuple[int, int]] = []  # (ans_start, a_len) per sample; used to slice logits/hidden states safely.

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
            # Padding is performed by the tokenizer to preserve model-specific pad semantics.
            cond_batch = tok.pad(cond_feats, return_tensors="pt").to(llm.input_device)
            uncond_batch = tok.pad(uncond_feats, return_tensors="pt").to(llm.input_device)

            out_c = model(**cond_batch, output_hidden_states=True, use_cache=False)
            out_u = model(**uncond_batch, output_hidden_states=True, use_cache=False)

            logits_c = out_c.logits
            logits_u = out_u.logits
            h_c_last = out_c.hidden_states[-1]
            h_u_last = out_u.hidden_states[-1]

            # Alignment invariant: batch dimension must match; otherwise indices/meta no longer correspond.
            if logits_c.size(0) != logits_u.size(0) or h_c_last.size(0) != h_u_last.size(0):
                if strict:
                    raise RuntimeError("Batch size mismatch between conditional/unconditional forward outputs.")
                # Deterministic batch-level fallback: preserve output length and ordering.
                results.extend(
                    [
                        {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []}
                        for _ in batch
                    ]
                )
                continue

            for i, (ans_start, a_len) in enumerate(meta):
                if a_len <= 0:
                    # Degenerate answer span -> do not emit NaNs (stable aggregation downstream).
                    results.append({"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []})
                    continue

                pred_start, pred_end = _prediction_slice(ans_start, a_len)
                if pred_start < 0 or pred_end <= pred_start:
                    # NOTE: potential issue: ans_start==0 implies pred_start=-1; caller/encode_qa should ensure answer follows a prompt token.
                    results.append({"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "emb_diff": 0.0, "grad_norm": 0.0, "g_vec": [], "e_vec": []})
                    continue

                # Slice only the answer-prediction logits; everything else is irrelevant to the primitives.
                lc = logits_c[i : i + 1, pred_start:pred_end, :]
                lu = logits_u[i : i + 1, pred_start:pred_end, :]

                # d_loss: KL(P(.|Q) || P(.|0)) over answer prediction positions.
                kl_tok = _kl_tokenwise_from_logits(lc, lu)
                d_loss = float(kl_tok.mean().item())

                # ce_loss: H(P(.|Q), P(.|0)) over answer prediction positions.
                ce_tok = _cross_entropy_tokenwise_from_logits(lc, lu)
                ce_loss = float(ce_tok.mean().item())

                # h_p: H(P(.|Q)) over answer prediction positions.
                h_tok = _entropy_tokenwise_from_logits(lc)
                h_p = float(h_tok.mean().item())

                # emb_diff: compare last-layer hidden states on the maximal common answer span (avoids OOB on padding/length mismatches).
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
        # Deterministic failure mode for strict=False: preserve list length and ordering to keep alignment with qa_pairs.
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
    Compute per-QA raw primitives; includes grad_norm via a backward pass on unconditional embeddings.
    chunk_size optionally accumulates KL over time to reduce peak memory (retain_graph=True per chunk).
    """
    try:
        model = llm.model
        model.eval()

        inputs_c, inputs_u, ans_start, a_len, _ = _encode_cond_uncond(llm, question, answer)
        if a_len <= 0:
            # Stable zero output on empty answer; prevents downstream NaNs/shape drift.
            return {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "grad_norm": 0.0, "emb_diff": 0.0, "g_vec": [], "e_vec": []}
        
        pred_start, pred_end = _prediction_slice(ans_start, a_len)
        if pred_start < 0 or pred_end <= pred_start:
            # NOTE: potential issue: invalid prediction slice implies misaligned ans_start/a_len w.r.t. causal shift.
            return {"d_loss": 0.0, "ce_loss": 0.0, "h_p": 0.0, "grad_norm": 0.0, "emb_diff": 0.0, "g_vec": [], "e_vec": []}

        # ------------------------------------------------------------------
        # (1) d_loss + (ce_loss) + (h_p) + (2) emb_diff (no grad)
        # ------------------------------------------------------------------
        with torch.no_grad():
            out_c = model(**inputs_c, output_hidden_states=True, use_cache=False)
            out_u = model(**inputs_u, output_hidden_states=True, use_cache=False)

            # Time alignment: slice logits that predict answer tokens (causal shift handled by _prediction_slice()).
            logits_c = out_c.logits[:, pred_start:pred_end, :]
            logits_u = out_u.logits[:, pred_start:pred_end, :]

            kl_tok = _kl_tokenwise_from_logits(logits_c, logits_u)  # (1, T_pred)
            d_loss = float(kl_tok.mean().item())

            ce_tok = _cross_entropy_tokenwise_from_logits(logits_c, logits_u)  # (1, T_pred)
            ce_loss = float(ce_tok.mean().item())

            h_tok = _entropy_tokenwise_from_logits(logits_c)  # (1, T_pred)
            h_p = float(h_tok.mean().item())

            # Hu-style "E": last-layer hidden state difference on answer token positions (not pooled CLS-style).
            h_c_last = out_c.hidden_states[-1]  # (1, seq_len_c, H)
            h_u_last = out_u.hidden_states[-1]  # (1, seq_len_u, H)

            seq_len_c = int(h_c_last.size(1))
            seq_len_u = int(h_u_last.size(1))
            L = _common_answer_len(seq_len_c, seq_len_u, ans_start, a_len)



            if L <= 0:
                # Deterministic fallback: no comparable answer span -> zero scalar and empty vector.
                emb_diff = 0.0
                e_vec = torch.zeros((0,), device=llm.input_device)
            else:
                h_c_ans = h_c_last[:, ans_start : ans_start + L, :]
                h_u_ans = h_u_last[:, ans_start : ans_start + L, :]

                # Scalar diagnostic: mean tokenwise L2 distance between conditional and unconditional hidden states.
                emb_diff = float(torch.norm(h_c_ans - h_u_ans, dim=-1).mean().item())

                # Fixed-size vector: mean over answer tokens to (H,), enabling downstream linear probes / aggregation.
                e_vec = (h_c_ans - h_u_ans).mean(dim=1).squeeze(0)   # shape (H,)

            # Detach conditional logits: gradients should flow only through unconditional embeddings for "G".
            logits_c_det = logits_c.detach()

        # ------------------------------------------------------------------
        # (3) grad_norm (KL-gradient wrt unconditional input embeddings)
        # ------------------------------------------------------------------
        input_embeddings = model.get_input_embeddings()

        # Gradients are taken w.r.t. unconditional *input embeddings* (not weights); detach to avoid accidental parameter grads.
        emb_u = input_embeddings(inputs_u["input_ids"]).detach()
        emb_u.requires_grad_(True)

        # Ensure a clean gradient state; avoids silent accumulation across calls in long-running processes.
        model.zero_grad(set_to_none=True)
        out_u_emb = model(
            inputs_embeds=emb_u,
            attention_mask=inputs_u.get("attention_mask", None),
            use_cache=False,
        )

        logits_u_emb = out_u_emb.logits[:, pred_start:pred_end, :]  # (1, T_pred_u, V)

        # Critical invariant: KL is computed token-aligned in time; mismatch implies invalid gradient feature.
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

        # KL(P(.|Q) || P(.|0)) with P fixed (detached); gradients propagate only through unconditional logits/embeddings.
        logp_c = torch.log_softmax(logits_c_det, dim=-1)
        p_c = logp_c.exp()
        logq_u = torch.log_softmax(logits_u_emb, dim=-1)
        kl_per_tok = torch.sum(p_c * (logp_c - logq_u), dim=-1)  # (1, T_pred)

        if chunk_size is None:
            # Single backward is simplest; use chunking only when necessary to avoid retain_graph overhead.
            loss = kl_per_tok.mean()
            loss.backward()
        else:
            # NOTE: potential issue: chunking uses repeated backward(retain_graph=True), trading memory for extra compute and potential slowdown.
            cs = int(chunk_size)
            T_pred = int(kl_per_tok.size(1))
            for s in range(0, T_pred, cs):
                e = min(T_pred, s + cs)
                loss_chunk = kl_per_tok[:, s:e].mean()
                loss_chunk.backward(retain_graph=True)

        if emb_u.grad is None:
            # Hard failure: grad_norm is undefined; do not silently report zeros under strict=True.
            raise RuntimeError("Gradient is None after backward() (unexpected).")

        # Compute grad feature over the maximal in-bounds answer span; mirrors emb_diff logic but only unconditional has grads.
        seq_len_u_emb = int(emb_u.size(1))
        # NOTE: unconditional/unconditional passed intentionally to reuse clamping logic without introducing new helpers.
        Lg = _common_answer_len(seq_len_u_emb, seq_len_u_emb, ans_start, a_len)

        if Lg <= 0:
            grad_norm = 0.0
            g_vec = torch.zeros((0,), device=llm.input_device)
        else:
            grad_ans = emb_u.grad[:, ans_start : ans_start + Lg, :]  # (1, Lg, H)
            
            # Scalar diagnostic: mean tokenwise gradient L2 norm (embedding space).
            grad_norm = float(grad_ans.norm(dim=-1).mean().item())

            # Fixed-size vector: mean over answer tokens to (H,), matching e_vec dimensionality.
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
        # strict=False provides a deterministic "no-signal" record to keep pipelines running and preserve alignment.
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