"""Bootstrap utilities for AUROC and paired AUROC differences with percentile CIs.

Inputs: binary labels `y` (0/1) and prediction scores (`score`, or `score_a`/`score_b`).
Outputs: point estimates plus percentile confidence intervals and (optionally) stored resample indices.
Implements stratified resampling (within-class) by default to keep class counts fixed per replicate.
Determinism: all sampling is driven by `BootstrapConfig.seed` via NumPy's Generator for reproducibility.
"""

# src/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap resampling and percentile CI construction."""
    B: int = 5000
    ci: float = 0.95
    seed: int = 42
    stratified: bool = True  # NOTE: potential issue: stratification assumes meaningful 0/1 labels only.


@dataclass(frozen=True)
class BootstrapResult:
    """AUROC point estimate with percentile CI and per-replicate AUROC samples (optionally indices)."""
    auc: float
    ci_low: float
    ci_high: float
    auc_samples: np.ndarray  # shape (n_valid,)
    indices: Optional[np.ndarray] = None  # shape (n_valid, N); indices[i] -> auc_samples[i] if stored
    n_valid: int = 0  # valid replicates retained (may be < n_total when stratified=False)
    n_total: int = 0  # requested replicates (cfg.B)
    ci: float = 0.95
    stratified: bool = True
    seed: int = 42


# ----------------------------
# Paired-diff bootstrap result with optional indices
# ----------------------------

@dataclass(frozen=True)
class BootstrapDiffResult:
    """Paired-bootstrap AUROC difference (AUC(A)-AUC(B)) with percentile CI and samples (optionally indices)."""
    diff: float
    ci_low: float
    ci_high: float
    diff_samples: np.ndarray  # shape (n_valid,)
    indices: Optional[np.ndarray] = None  # shape (n_valid, N); indices[i] -> diff_samples[i] if stored
    n_valid: int = 0  # valid replicates retained (may be < n_total when stratified=False)
    n_total: int = 0  # requested replicates (cfg.B)
    ci: float = 0.95
    stratified: bool = True
    seed: int = 42


def _validate_bootstrap_inputs(
    y: np.ndarray,
    score: np.ndarray,
    cfg: BootstrapConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    # Normalize to 1D numeric arrays; AUROC expects matching lengths and finite scores.
    y = np.asarray(y).astype(int).reshape(-1)
    score = np.asarray(score).astype(float).reshape(-1)

    if y.shape[0] != score.shape[0]:
        raise ValueError("y and score must have same length.")
    if y.shape[0] == 0:
        raise ValueError("Empty inputs.")
    if cfg.B <= 0:
        raise ValueError(f"B must be > 0, got {cfg.B}")
    if not (0.0 < float(cfg.ci) < 1.0):
        raise ValueError(f"ci must be in (0,1), got {cfg.ci}")
    if np.unique(y).size < 2:
        # AUROC is undefined if the dataset has only one class (including after preprocessing).
        raise ValueError("AUROC undefined: y has only one class.")
    if not np.all(np.isfinite(score)):
        # Avoid silent propagation of NaN/Inf through sklearn metric and quantiles.
        raise ValueError("score contains non-finite values (NaN/Inf).")

    return y, score


def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return resampling indices by bootstrapping within each class, preserving original class counts."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = idx0.shape[0]
    n1 = idx1.shape[0]

    # Stratified resampling: guarantees both classes present given original has both classes.
    s0 = rng.choice(idx0, size=n0, replace=True)
    s1 = rng.choice(idx1, size=n1, replace=True)

    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)  # Mix classes to avoid any implicit ordering assumptions downstream.
    return idx


def _plain_bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    # Plain bootstrap over all observations; may yield single-class samples when classes are imbalanced.
    return rng.integers(0, n, size=n, endpoint=False)


def bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    cfg: BootstrapConfig,
    *,
    store_indices: bool = False,
) -> BootstrapResult:
    """Compute bootstrap AUROC samples and a percentile CI (optionally storing resample indices)."""
    y, score = _validate_bootstrap_inputs(y, score, cfg)
    n = y.shape[0]

    # Point estimate computed on the full sample (not a bootstrap replicate).
    point_auc = float(roc_auc_score(y, score))

    # Deterministic sampling given cfg.seed; no global RNG state is used.
    rng = np.random.default_rng(int(cfg.seed))

    auc_list: list[float] = []
    idx_list: list[np.ndarray] = []

    for _ in range(int(cfg.B)):
        if cfg.stratified:
            idx = _stratified_bootstrap_indices(y, rng)
            # Defined by construction: each replicate contains both classes if original does.
            auc_val = float(roc_auc_score(y[idx], score[idx]))
            auc_list.append(auc_val)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))
        else:
            idx = _plain_bootstrap_indices(n, rng)
            y_b = y[idx]
            if np.unique(y_b).size < 2:
                # Skip invalid replicates to avoid undefined AUROC; reduces effective sample size (n_valid).
                continue
            auc_val = float(roc_auc_score(y_b, score[idx]))
            auc_list.append(auc_val)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))

    if len(auc_list) == 0:
        # All replicates invalid implies CI cannot be estimated; typically class imbalance + stratified=False.
        raise RuntimeError(
            "Bootstrap produced 0 valid replicates (likely due to extreme class imbalance with stratified=False)."
        )

    auc_samples = np.asarray(auc_list, dtype=np.float64)

    # Percentile CI on the bootstrap distribution; alpha determined by cfg.ci.
    alpha = (1.0 - float(cfg.ci)) / 2.0
    low = float(np.quantile(auc_samples, alpha))
    high = float(np.quantile(auc_samples, 1.0 - alpha))

    indices = None
    if store_indices:
        # Invariant: len(idx_list) == len(auc_list) because indices are stored only for valid replicates.
        indices = np.stack(idx_list, axis=0).astype(np.int32, copy=False)

    return BootstrapResult(
        auc=point_auc,
        ci_low=low,
        ci_high=high,
        auc_samples=auc_samples,
        indices=indices,
        n_valid=int(auc_samples.shape[0]),
        n_total=int(cfg.B),
        ci=float(cfg.ci),
        stratified=bool(cfg.stratified),
        seed=int(cfg.seed),
    )


def bootstrap_auc_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cfg: BootstrapConfig,
) -> Tuple[float, float, float]:
    """
    Paired bootstrap percentile CI for AUROC difference, using shared indices per replicate.

    Uses the same bootstrap indices for A and B to preserve within-sample correlation of AUROC estimates.
    Returns (point_diff, ci_low, ci_high) where point_diff = AUC(score_a) - AUC(score_b).
    """
    # Normalize to 1D numeric arrays; keep explicit validation here to avoid cross-function coupling.
    y = np.asarray(y).astype(int).reshape(-1)
    score_a = np.asarray(score_a).astype(float).reshape(-1)
    score_b = np.asarray(score_b).astype(float).reshape(-1)

    if y.shape[0] != score_a.shape[0] or y.shape[0] != score_b.shape[0]:
        raise ValueError("y, score_a, score_b must have same length.")
    if y.shape[0] == 0:
        raise ValueError("Empty inputs.")
    if cfg.B <= 0:
        raise ValueError(f"B must be > 0, got {cfg.B}")
    if not (0.0 < float(cfg.ci) < 1.0):
        raise ValueError(f"ci must be in (0,1), got {cfg.ci}")
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: y has only one class.")
    if not (np.all(np.isfinite(score_a)) and np.all(np.isfinite(score_b))):
        raise ValueError("Scores contain non-finite values (NaN/Inf).")

    n = y.shape[0]
    # Sign convention: positive values indicate score_a yields higher AUROC than score_b.
    diff_point = float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b))

    rng = np.random.default_rng(int(cfg.seed))
    diffs: list[float] = []

    for _ in range(int(cfg.B)):
        if cfg.stratified:
            idx = _stratified_bootstrap_indices(y, rng)
            # Paired bootstrap: same idx applied to both score vectors.
            d = float(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
            diffs.append(d)
        else:
            idx = _plain_bootstrap_indices(n, rng)
            y_b = y[idx]
            if np.unique(y_b).size < 2:
                # Skip invalid replicates; effective n_valid can be much smaller than cfg.B.
                continue
            d = float(roc_auc_score(y_b, score_a[idx]) - roc_auc_score(y_b, score_b[idx]))
            diffs.append(d)

    if len(diffs) == 0:
        raise RuntimeError(
            "Bootstrap diff produced 0 valid replicates (likely due to extreme class imbalance with stratified=False)."
        )

    diffs_arr = np.asarray(diffs, dtype=np.float64)

    alpha = (1.0 - float(cfg.ci)) / 2.0
    low = float(np.quantile(diffs_arr, alpha))
    high = float(np.quantile(diffs_arr, 1.0 - alpha))

    return diff_point, low, high


# ----------------------------
# Paired diff bootstrap with optional stored indices
# ----------------------------

def bootstrap_auc_diff_with_indices(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cfg: BootstrapConfig,
    *,
    store_indices: bool = False,
) -> BootstrapDiffResult:
    """Paired-bootstrap AUROC difference with percentile CI and optional resample index storage."""
    y = np.asarray(y).astype(int).reshape(-1)
    score_a = np.asarray(score_a).astype(float).reshape(-1)
    score_b = np.asarray(score_b).astype(float).reshape(-1)

    if y.shape[0] != score_a.shape[0] or y.shape[0] != score_b.shape[0]:
        raise ValueError("y, score_a, score_b must have same length.")
    if y.shape[0] == 0:
        raise ValueError("Empty inputs.")
    if cfg.B <= 0:
        raise ValueError(f"B must be > 0, got {cfg.B}")
    if not (0.0 < float(cfg.ci) < 1.0):
        raise ValueError(f"ci must be in (0,1), got {cfg.ci}")
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: y has only one class.")
    if not (np.all(np.isfinite(score_a)) and np.all(np.isfinite(score_b))):
        raise ValueError("Scores contain non-finite values (NaN/Inf).")

    n = y.shape[0]
    diff_point = float(roc_auc_score(y, score_a) - roc_auc_score(y, score_b))

    rng = np.random.default_rng(int(cfg.seed))

    diffs: list[float] = []
    idx_list: list[np.ndarray] = []

    for _ in range(int(cfg.B)):
        if cfg.stratified:
            idx = _stratified_bootstrap_indices(y, rng)
            # Paired bootstrap: shared idx preserves correlation between AUROC estimates.
            d = float(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
            diffs.append(d)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))
        else:
            idx = _plain_bootstrap_indices(n, rng)
            y_b = y[idx]
            if np.unique(y_b).size < 2:
                continue
            d = float(roc_auc_score(y_b, score_a[idx]) - roc_auc_score(y_b, score_b[idx]))
            diffs.append(d)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))

    if len(diffs) == 0:
        raise RuntimeError(
            "Bootstrap diff produced 0 valid replicates (likely due to extreme class imbalance with stratified=False)."
        )

    diffs_arr = np.asarray(diffs, dtype=np.float64)

    alpha = (1.0 - float(cfg.ci)) / 2.0
    low = float(np.quantile(diffs_arr, alpha))
    high = float(np.quantile(diffs_arr, 1.0 - alpha))

    indices = None
    if store_indices:
        # Invariant: len(idx_list) == len(diffs) because indices are stored only for valid replicates.
        indices = np.stack(idx_list, axis=0).astype(np.int32, copy=False)

    return BootstrapDiffResult(
        diff=diff_point,
        ci_low=low,
        ci_high=high,
        diff_samples=diffs_arr,
        indices=indices,
        n_valid=int(diffs_arr.shape[0]),
        n_total=int(cfg.B),
        ci=float(cfg.ci),
        stratified=bool(cfg.stratified),
        seed=int(cfg.seed),
    )