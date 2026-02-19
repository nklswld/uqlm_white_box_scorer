# src/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapConfig:
    """
    Bootstrap configuration.

    B:
      Number of bootstrap replicates.

    ci:
      Confidence level for percentile CI (e.g., 0.95).

    seed:
      RNG seed.

    stratified:
      If True: resample WITHIN each class to preserve class counts (recommended for AUROC).
      If False: plain bootstrap over all indices (can produce single-class samples; will be skipped).
    """
    B: int = 5000
    ci: float = 0.95
    seed: int = 42
    stratified: bool = True


@dataclass(frozen=True)
class BootstrapResult:
    """
    Result of bootstrap AUROC with percentile CI.

    auc_samples and (optionally) indices are aligned: auc_samples[i] was computed on indices[i].
    If stratified=False, invalid (single-class) replicates are dropped; see n_valid.
    """
    auc: float
    ci_low: float
    ci_high: float
    auc_samples: np.ndarray  # shape (n_valid,)
    indices: Optional[np.ndarray] = None  # shape (n_valid, N)
    n_valid: int = 0
    n_total: int = 0
    ci: float = 0.95
    stratified: bool = True
    seed: int = 42


# ----------------------------
# Paired-diff bootstrap result with optional indices
# ----------------------------

@dataclass(frozen=True)
class BootstrapDiffResult:
    """
    Result of paired bootstrap for AUROC difference: AUC(A) - AUC(B).

    diff_samples and (optionally) indices are aligned: diff_samples[i] was computed on indices[i].
    If stratified=False, invalid (single-class) replicates are dropped; see n_valid.
    """
    diff: float
    ci_low: float
    ci_high: float
    diff_samples: np.ndarray  # shape (n_valid,)
    indices: Optional[np.ndarray] = None  # shape (n_valid, N)
    n_valid: int = 0
    n_total: int = 0
    ci: float = 0.95
    stratified: bool = True
    seed: int = 42


def _validate_bootstrap_inputs(
    y: np.ndarray,
    score: np.ndarray,
    cfg: BootstrapConfig,
) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError("AUROC undefined: y has only one class.")
    if not np.all(np.isfinite(score)):
        raise ValueError("score contains non-finite values (NaN/Inf).")

    return y, score


def _stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Stratified resampling indices: sample with replacement within each class,
    keeping class counts equal to the original, then shuffle.
    """
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = idx0.shape[0]
    n1 = idx1.shape[0]

    s0 = rng.choice(idx0, size=n0, replace=True)
    s1 = rng.choice(idx1, size=n1, replace=True)

    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)
    return idx


def _plain_bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)


def bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    cfg: BootstrapConfig,
    *,
    store_indices: bool = False,
) -> BootstrapResult:
    """
    Bootstrap AUROC with percentile confidence interval.

    - If cfg.stratified=True, each bootstrap sample preserves class counts and AUROC is always defined.
    - If cfg.stratified=False, some samples may contain only one class; those replicates are skipped.

    Returns BootstrapResult with auc_samples aligned to indices (if stored).
    """
    y, score = _validate_bootstrap_inputs(y, score, cfg)
    n = y.shape[0]

    point_auc = float(roc_auc_score(y, score))

    rng = np.random.default_rng(int(cfg.seed))

    auc_list: list[float] = []
    idx_list: list[np.ndarray] = []

    for _ in range(int(cfg.B)):
        if cfg.stratified:
            idx = _stratified_bootstrap_indices(y, rng)
            # Defined by construction
            auc_val = float(roc_auc_score(y[idx], score[idx]))
            auc_list.append(auc_val)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))
        else:
            idx = _plain_bootstrap_indices(n, rng)
            y_b = y[idx]
            if np.unique(y_b).size < 2:
                continue
            auc_val = float(roc_auc_score(y_b, score[idx]))
            auc_list.append(auc_val)
            if store_indices:
                idx_list.append(idx.astype(np.int32, copy=False))

    if len(auc_list) == 0:
        raise RuntimeError(
            "Bootstrap produced 0 valid replicates (likely due to extreme class imbalance with stratified=False)."
        )

    auc_samples = np.asarray(auc_list, dtype=np.float64)

    alpha = (1.0 - float(cfg.ci)) / 2.0
    low = float(np.quantile(auc_samples, alpha))
    high = float(np.quantile(auc_samples, 1.0 - alpha))

    indices = None
    if store_indices:
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
    Bootstrap CI for AUROC difference: AUC(A) - AUC(B).

    Important: uses SHARED bootstrap indices ("paired bootstrap") so that the
    correlation between the two AUC estimates is preserved.

    CI method: percentile bootstrap CI over the bootstrap distribution of differences.
    """
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

    for _ in range(int(cfg.B)):
        if cfg.stratified:
            idx = _stratified_bootstrap_indices(y, rng)
            d = float(roc_auc_score(y[idx], score_a[idx]) - roc_auc_score(y[idx], score_b[idx]))
            diffs.append(d)
        else:
            idx = _plain_bootstrap_indices(n, rng)
            y_b = y[idx]
            if np.unique(y_b).size < 2:
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
    """
    Paired bootstrap for AUROC difference: AUC(A) - AUC(B), with OPTIONAL index storage.

    - Uses SHARED bootstrap indices per replicate (paired bootstrap).
    - If cfg.stratified=True, each replicate preserves class counts and AUROC is always defined.
    - If cfg.stratified=False, replicates that contain only one class are skipped.

    Returns BootstrapDiffResult with diff_samples aligned to indices (if stored).
    """
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