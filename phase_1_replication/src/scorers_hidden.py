# src/scorers_hidden.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Public Protocol (dataset-agnostic)
# =============================================================================

class QAProto(Protocol):
    """
    Minimal interface expected by the hidden-state feature extractor.
    Intentionally dataset-agnostic.
    """
    question: str
    model_answer: str


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class HiddenFeatureConfig:
    """
    Feature extraction config for hidden-state methods.

    layers:
        Transformer layer indices to use (hidden_states indexing).
        Negative indices follow Python semantics (hidden_states[-1] = last).

    pooling:
        How to pool token-level hidden states into a fixed vector.
        - "mean_answer": mean over answer tokens only
        - "last_answer": last answer token vector
        - "mean_all":    mean over all *real* tokens (mask-aware)

    normalize:
        If True, L2-normalize the pooled vector(s) per sample.

    batch_size:
        Batch size for feature extraction forward passes (output_hidden_states=True).
        With 48GB VRAM you can often run 16–64 depending on model + seq length.
    """
    layers: Tuple[int, ...] = (-1,)
    pooling: str = "mean_answer"
    normalize: bool = False
    batch_size: int = 16


@dataclass(frozen=True)
class HiddenScorerConfig:
    """
    Paper-ready supervised scorer configuration for hidden-state features.

    - OOF probabilities via Stratified K-Fold
    - StandardScaler inside Pipeline to avoid leakage
    - fixed seed for reproducibility
    """
    n_splits: int = 5
    random_state: int = 42
    max_iter: int = 2000
    C: float = 1.0
    solver: str = "lbfgs"
    class_weight: Optional[str] = None  # e.g., "balanced"


# =============================================================================
# Helpers
# =============================================================================

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


def _validate_layers(layer_indices: Tuple[int, ...], n_hidden_states: int) -> None:
    """
    Validate that each layer index is valid for hidden_states length.
    hidden_states includes embeddings + each transformer layer; HF usually returns len = n_layers + 1.
    """
    for li in layer_indices:
        if not (-n_hidden_states <= li < n_hidden_states):
            raise ValueError(
                f"Invalid layer index {li} for hidden_states length={n_hidden_states}. "
                f"Valid range: [{-n_hidden_states}, {n_hidden_states - 1}]"
            )


def _last_real_token_end(attn_row: torch.Tensor) -> int:
    """
    Compute end index (exclusive) of real tokens in PADDED space using attention_mask row.
    Returns 0 if no real tokens.
    """
    idx = torch.nonzero(attn_row, as_tuple=False).flatten()
    if idx.numel() == 0:
        return 0
    return int(idx[-1].item()) + 1


def _pool_hidden_masked(
    h_row: torch.Tensor,
    ans_start: int,
    attn_row: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """
    Pool a single-sample hidden-state row into a single vector.

    Paper-ready behavior (Must-Fix 2):
    - If ans_start >= end_real (invalid/empty answer span), DO NOT fallback.
      Raise instead, so invalid spans cannot silently produce "mean_all" / "last token" features.
    """
    T, D = h_row.shape
    end_real = _last_real_token_end(attn_row)  # exclusive
    if end_real <= 0:
        return torch.zeros((D,), device=h_row.device, dtype=h_row.dtype)

    real_mask = attn_row.bool()

    if pooling == "mean_all":
        h_real = h_row[real_mask, :]
        if h_real.numel() == 0:
            return torch.zeros((D,), device=h_row.device, dtype=h_row.dtype)
        return h_real.mean(dim=0)

    # Answer-aware poolings (STRICT: no fallback)
    if ans_start >= end_real:
        raise ValueError(
            f"Invalid answer span for answer-aware pooling: ans_start={ans_start} >= end_real={end_real}. "
            "Do not fallback to mean_all/last token; skip this sample upstream."
        )

    ans_span = h_row[ans_start:end_real, :]
    if ans_span.size(0) == 0:
        raise ValueError(
            f"Empty answer span for answer-aware pooling: ans_start={ans_start}, end_real={end_real}. "
            "Do not fallback to mean_all/last token; skip this sample upstream."
        )

    if pooling == "mean_answer":
        return ans_span.mean(dim=0)

    if pooling == "last_answer":
        return ans_span[-1, :]

    raise ValueError(f"Unknown pooling strategy: {pooling}")


# =============================================================================
# Feature extraction (batched, paper-ready)
# =============================================================================

def build_hidden_feature_matrix(
    examples: List[QAProto],
    llm,
    *,
    feature_cfg: HiddenFeatureConfig = HiddenFeatureConfig(),
    strict: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build hidden-state feature matrix from frozen model answers.

    Robust behavior:
    - skips empty answers
    - skips examples where answer_start_idx is invalid (>= end_real)
    - reports meta/audit stats
    """
    pooling = feature_cfg.pooling
    layers = tuple(feature_cfg.layers)
    bs = int(feature_cfg.batch_size)
    if bs <= 0:
        raise ValueError(f"batch_size must be positive, got {bs}")

    feats: List[np.ndarray] = []
    kept: List[int] = []

    skipped_empty = 0
    skipped_span = 0
    skipped_other = 0
    dropped_empty_idx: List[int] = []
    dropped_span_idx: List[int] = []
    dropped_other_idx: List[int] = []

    valid_pairs: List[Tuple[int, str, str]] = []
    for i, ex in enumerate(examples):
        ans = getattr(ex, "model_answer", None)
        if not ans or not isinstance(ans, str) or not ans.strip():
            skipped_empty += 1
            dropped_empty_idx.append(i)
            continue
        valid_pairs.append((i, ex.question, ans))

    if not valid_pairs:
        raise ValueError("No valid examples (all answers empty).")

    llm.model.eval()

    with torch.no_grad():
        for s in range(0, len(valid_pairs), bs):
            batch = valid_pairs[s : s + bs]
            idxs = [b[0] for b in batch]
            qs = [b[1] for b in batch]
            ans = [b[2] for b in batch]

            try:
                enc_b = llm.encode_qa_batch(qs, ans)
                out = llm.forward(
                    input_ids=enc_b.input_ids,
                    attention_mask=enc_b.attention_mask,
                    output_hidden_states=True,
                )
                hs = out.hidden_states
                if hs is None:
                    raise RuntimeError("Model did not return hidden_states (output_hidden_states=True failed).")

                n_hs = len(hs)
                _validate_layers(layers, n_hs)

                B, T, D = hs[-1].shape

                for bi in range(B):
                    ans_start = int(enc_b.answer_start_idx[bi].item())
                    attn_row = enc_b.attention_mask[bi]
                    end_real = _last_real_token_end(attn_row)

                    if ans_start >= end_real or end_real == 0:
                        skipped_span += 1
                        dropped_span_idx.append(idxs[bi])
                        continue

                    pooled_parts: List[torch.Tensor] = []
                    for layer_idx in layers:
                        h_layer = hs[layer_idx][bi]
                        pooled = _pool_hidden_masked(
                            h_row=h_layer,
                            ans_start=ans_start,
                            attn_row=attn_row,
                            pooling=pooling,
                        )
                        pooled_parts.append(pooled)

                    feat = torch.cat(pooled_parts, dim=0)
                    feat_np = feat.detach().to(torch.float32).cpu().numpy()

                    if feat_np.ndim != 1:
                        raise ValueError(f"Expected 1D feature vector, got {feat_np.shape} at idx={idxs[bi]}")

                    if feature_cfg.normalize:
                        feat_np = _l2_normalize(feat_np)

                    feats.append(feat_np)
                    kept.append(idxs[bi])

            except Exception:
                if strict:
                    raise
                skipped_other += len(batch)
                dropped_other_idx.extend(idxs)

    if not feats:
        raise ValueError("No hidden features extracted (all valid answers had invalid spans or failures).")

    X = np.stack(feats, axis=0).astype(np.float64, copy=False)
    kept_idx = np.asarray(kept, dtype=int)

    meta: Dict[str, Any] = {
        "layers": list(layers),
        "pooling": pooling,
        "normalize": bool(feature_cfg.normalize),
        "batch_size": int(bs),
        "n_total": int(len(examples)),
        "n_after_empty_filter": int(len(valid_pairs)),
        "n_kept": int(len(kept)),
        "skipped_empty_answer": int(skipped_empty),
        "skipped_invalid_answer_span": int(skipped_span),
        "skipped_other": int(skipped_other),
        "dropped_idx": {
            "empty_answer": dropped_empty_idx,
            "invalid_span": dropped_span_idx,
            "other": dropped_other_idx,
        },
        "feature_dim": int(X.shape[1]),
    }

    return X, kept_idx, meta


# =============================================================================
# Supervised OOF training (hidden scorer)
# =============================================================================

def _compute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    s = np.asarray(s).reshape(-1).astype(float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("AUROC: y and scores must have same length.")
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: y contains only one class.")
    return float(roc_auc_score(y, s))


def _make_model(cfg: HiddenScorerConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=cfg.max_iter,
                    C=cfg.C,
                    solver=cfg.solver,
                    class_weight=cfg.class_weight,
                ),
            ),
        ]
    )


def train_hidden_scorer_oof(
    X: np.ndarray,
    y: np.ndarray,
    cfg: HiddenScorerConfig,
    *,
    return_final_model: bool = True,
) -> Dict[str, Any]:
    """
    Train a scorer with Out-of-Fold (OOF) probabilities.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(int)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if np.unique(y).size < 2:
        raise ValueError("y contains only one class; cannot train/evaluate AUROC.")

    skf = StratifiedKFold(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.random_state,
    )

    oof_scores = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=np.float64)
    fold_indices: List[Dict[str, Any]] = []

    for fold_id, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = _make_model(cfg)
        model.fit(X[tr], y[tr])

        probs = model.predict_proba(X[te])[:, 1]
        oof_scores[te] = probs

        fold_indices.append(
            {
                "fold": int(fold_id),
                "train_idx": tr.astype(int).tolist(),
                "test_idx": te.astype(int).tolist(),
                "n_train": int(tr.shape[0]),
                "n_test": int(te.shape[0]),
                "pos_train": int(y[tr].sum()),
                "pos_test": int(y[te].sum()),
            }
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("OOF scoring failed: some samples were never assigned a score.")

    auroc_oof = _compute_auroc(y, oof_scores)

    out: Dict[str, Any] = {
        "oof_scores": oof_scores,
        "auroc_oof": float(auroc_oof),
        "fold_indices": fold_indices,
    }

    if return_final_model:
        final_model = _make_model(cfg)
        final_model.fit(X, y)
        out["final_model"] = final_model

    return out


# =============================================================================
# Single Source of Truth OOF wrapper
# =============================================================================

def oof_logreg_scores(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    max_iter: int = 4000,
    C: float = 1.0,
    solver: str = "lbfgs",
    class_weight: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Minimal wrapper to provide the exact (scores, folds) API previously implemented in run_phase1_truthfulqa.py,
    but now centralized here (Single Source of Truth).

    Returns:
      scores: (n,) OOF probabilities for class 1
      folds:  list of per-fold summaries (counts only; no indices) for manifest logging
    """
    cfg = HiddenScorerConfig(
        n_splits=int(n_splits),
        random_state=int(seed),
        max_iter=int(max_iter),
        C=float(C),
        solver=str(solver),
        class_weight=class_weight,
    )

    res = train_hidden_scorer_oof(X, y, cfg, return_final_model=False)
    scores = res["oof_scores"]

    folds: List[Dict[str, Any]] = []
    for fd in res["fold_indices"]:
        folds.append(
            {
                "fold": int(fd["fold"]),
                "n_train": int(fd["n_train"]),
                "n_test": int(fd["n_test"]),
                "y_train_pos": int(fd["pos_train"]),
                "y_test_pos": int(fd["pos_test"]),
            }
        )

    return scores, folds