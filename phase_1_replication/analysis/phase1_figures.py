# phase1_figures.py
# -*- coding: utf-8 -*-
"""
Phase 1 figure generator for TruthfulQA (frozen-answer setting).

Loads per-example results from a JSONL file plus an authoritative run manifest (JSON),
then renders publication-ready PDF figures (AUROC/CIs, ROC, score distributions, overlap,
rank correlations, and targeted mismatch diagnostics).
Key inputs: --results (per-example JSONL), --manifest (run manifest with AUROC/bootstrap CI).
Key outputs: PDF figures written to --outdir (one file per figure).
Determinism: fully deterministic given fixed input files; AUROC/CIs are read from the manifest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

# Seaborn is used for high-level statistical plots (KDE/box/heatmap/scatter) while still
# deferring typography/layout to Matplotlib rcParams for consistent publication styling.
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve


# -----------------------------
# Global styling (Phase-2-like)
# -----------------------------
def apply_phase2_plot_style(font_scale: float = 1.2) -> None:
    """Set shared rcParams for Phase 1/2 visual consistency and PDF-safe font embedding."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": int(12 * font_scale),
            "axes.titlesize": int(13 * font_scale),
            "axes.labelsize": int(12 * font_scale),
            "xtick.labelsize": int(11 * font_scale),
            "ytick.labelsize": int(11 * font_scale),
            "legend.fontsize": int(11 * font_scale),
            "legend.title_fontsize": int(11 * font_scale),
            "figure.titlesize": int(14.5 * font_scale),
            "axes.titlepad": 12,
            # Print/PDF legibility
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            # Vector-friendly fonts (avoid Type-3 fonts in PDF)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif",
            # Slightly cleaner axes
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# -----------------------------
# Constants / naming
# -----------------------------
# Binary label convention used throughout: 1 => hallucinated, 0 => not hallucinated.
LABEL_COL: str = "hallucinated"

# Standard bar width used across the project for consistent visual density.
BAR_WIDTH: float = 0.65

# Display-name -> column in results DF (flattened JSON keys).
SCORE_COLS: Dict[str, str] = {
    "Hidden-state probe (OOF)": "scores.hidden_probe_oof",
    "LNTP (hallucination score)": "scores.lntp_uncertainty",
    "EGH probe (OOF)": "scores.egh_probe_oof",
    "MTP (hallucination score)": "scores.mtp_uncertainty",
}
COL_TO_NAME: Dict[str, str] = {v: k for k, v in SCORE_COLS.items()}

# Manifest keys for AUROC + bootstrap blocks (authoritative reporting source).
MANIFEST_KEYS: Dict[str, str] = {
    "Hidden-state probe (OOF)": "hidden_probe_oof",
    "LNTP (hallucination score)": "lntp_uncertainty",
    "MTP (hallucination score)": "mtp_uncertainty",
    "EGH probe (OOF)": "egh_probe_oof",
}

# Preferred display order (only keep those present).
DEFAULT_ORDER: List[str] = [
    "LNTP (hallucination score)",
    "MTP (hallucination score)",
    "EGH probe (OOF)",
    "Hidden-state probe (OOF)",
]

SCORER_SHORT = {
    "EGH probe (OOF)": "EGH",
    "LNTP (hallucination score)": "LNTP",
    "MTP (hallucination score)": "MTP",
    "Hidden-state probe (OOF)": "Hidden",
}

def short_name(name: str) -> str:
    """Stable short label for plotting; falls back to full name if unknown."""
    return SCORER_SHORT.get(name, name)

# -----------------------------
# I/O helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    """Create directory (and parents) if missing; no-op if already present."""
    path.mkdir(parents=True, exist_ok=True)


def safe_savefig(fig: plt.Figure, outpath: Path, dpi: int = 300) -> None:
    """Write a single PDF figure with tight bounding box for publication export."""
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved: {outpath}")


def load_results_jsonl(path: Path) -> pd.DataFrame:
    """Load per-example JSONL results and return a flattened DataFrame (one row per example)."""
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in results JSONL: {path}")
    return pd.json_normalize(rows)


def load_manifest(path: Path) -> dict:
    """Load run manifest JSON (authoritative AUROC/bootstrap CI source)."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Computation helpers
# -----------------------------
def compute_auroc_from_results(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute AUROC from per-example scores (diagnostic only; manifest is authoritative)."""
    missing_cols = [c for c in SCORE_COLS.values() if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected score columns: {missing_cols}")

    # NOTE: potential issue: AUROC assumes score polarity where higher scores imply "more hallucinated".
    y = df[LABEL_COL].values.astype(int)

    rows = []
    for name, col in SCORE_COLS.items():
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(s)  # drop NaNs/Infs to avoid implicit failures in sklearn metrics
        if mask.sum() == 0:
            continue
        auc = float(roc_auc_score(y[mask], s[mask]))
        rows.append({"Scorer": name, "AUROC": auc})

    out = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    return out


def build_ci_table_from_manifest(manifest: dict) -> pd.DataFrame:
    """Extract AUROC and bootstrap CI bounds from the manifest into a plotting-ready table."""
    auroc = manifest["scores"]["auroc"]
    boot = manifest["scores"]["bootstrap"]

    rows = []
    for label, k in MANIFEST_KEYS.items():
        if (k not in auroc) or (k not in boot):
            # Single warning per missing scorer to prevent silent omission from reported figures.
            print(f"[WARN] Missing in manifest: {k} (skipping)")
            continue
        rows.append(
            {
                "Scorer": label,
                "AUROC": float(auroc[k]),
                "CI_low": float(boot[k]["ci_low"]),
                "CI_high": float(boot[k]["ci_high"]),
            }
        )

    df_ci = pd.DataFrame(rows).sort_values("AUROC", ascending=False).reset_index(drop=True)
    # Precompute asymmetric errors to match typical bootstrap CI reporting.
    df_ci["err_low"] = df_ci["AUROC"] - df_ci["CI_low"]
    df_ci["err_high"] = df_ci["CI_high"] - df_ci["AUROC"]
    return df_ci


def ordered_subset(items: List[str], available: List[str]) -> List[str]:
    """Keep the relative order of `items` but only return those present in `available`."""
    return [x for x in items if x in set(available)]


# -----------------------------
# Figure finishing helper
# -----------------------------
def finalize_suptitle(fig: plt.Figure, title: str, *, y: float = 0.99, top_rect: float = 0.96) -> None:
    """Apply standardized suptitle + tight_layout headroom (shared across Phase 1/2 figures)."""
    fig.suptitle(title, y=y)
    fig.tight_layout(rect=[0, 0, 1, top_rect])


# -----------------------------
# Plotting functions
# -----------------------------
def plot_fig1_auroc_bar(ci_df: pd.DataFrame, outdir: Path) -> None:
    """Figure 1: AUROC comparison (horizontal bars with manifest bootstrap CI)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    y_pos = np.arange(len(ci_df))
    ax.barh(
        y_pos,
        ci_df["AUROC"].values,
        xerr=[ci_df["err_low"].values, ci_df["err_high"].values],
        align="center",
        alpha=0.9,
        capsize=4,
        height=BAR_WIDTH,  # slightly slimmer bars for denser label packing
    )

    # Value labels: offset in screen-space to avoid overlap with CI whiskers (stable across x-scales).
    for i, val in enumerate(ci_df["AUROC"].values):
        ax.annotate(
            f"{val:.3f}",
            xy=(val + 0.005, i),
            xytext=(0, 3),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=int(mpl.rcParams["xtick.labelsize"] * 0.85),
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),  # ensures legibility over CI lines
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_name(s) for s in ci_df["Scorer"].tolist()])
    ax.invert_yaxis()  # invariant: highest AUROC displayed at the top

    # Reference baseline for random discrimination (binary AUROC = 0.5).
    ax.axvline(0.5, linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel("AUROC (higher = better hallucination discrimination)")
    ax.set_xlim(0.4, 0.8)  # fixed range improves cross-run visual comparability
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    finalize_suptitle(fig, "AUROC Comparison (TruthfulQA, Frozen Answers)", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "fig1_auroc_bar_comparison.pdf")
    plt.close(fig)


def plot_fig2_bootstrap_ci(ci_df: pd.DataFrame, outdir: Path) -> None:
    """Figure 2: AUROC with bootstrap 95% CI (interval plot, fixed scorer order)."""
    order = ordered_subset(DEFAULT_ORDER, ci_df["Scorer"].tolist())
    # NOTE: potential issue: .loc[order] assumes all requested scorers exist; ordered_subset enforces that.
    ci_df_plot = ci_df.set_index("Scorer").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    ypos = np.arange(len(ci_df_plot))
    x = ci_df_plot["AUROC"].values
    lo = ci_df_plot["CI_low"].values
    hi = ci_df_plot["CI_high"].values

    ax.hlines(ypos, lo, hi, linewidth=3)
    ax.plot(x, ypos, "o")

    ax.set_yticks(ypos)
    ax.set_yticklabels([short_name(s) for s in ci_df_plot["Scorer"].tolist()])
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0.40, 0.80)  # fixed axis supports side-by-side figure comparisons
    ax.invert_yaxis()

    ax.set_xlabel("AUROC (higher = better hallucination discrimination)")
    ax.set_ylabel("")

    finalize_suptitle(fig, "AUROC with Bootstrap 95% CI (TruthfulQA, Frozen Answers)", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "fig2_auroc_bootstrap_ci.pdf")
    plt.close(fig)


def plot_fig3_roc_overlay(df: pd.DataFrame, outdir: Path) -> None:
    """Figure 3: ROC overlays for the preferred scorer subset (NaNs dropped per scorer)."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    y = df[LABEL_COL].values.astype(int)

    roc_show = ordered_subset(DEFAULT_ORDER, list(SCORE_COLS.keys()))
    for name in roc_show:
        col = SCORE_COLS[name]
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(s)  # per-scorer filtering preserves maximum usable samples
        if mask.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y[mask], s[mask])
        auc = roc_auc_score(y[mask], s[mask])
        ax.plot(fpr, tpr, label=f"{short_name(name)} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)

    finalize_suptitle(fig, "ROC Curves (Phase 1, TruthfulQA)", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "fig3_roc_overlay.pdf")
    plt.close(fig)


def plot_fig4_score_distributions(df: pd.DataFrame, outdir: Path) -> None:
    """Figure 4: score distributions by label (boxplots; NaNs/Infs removed)."""
    # Melt to long form: each row becomes (qid, label, scorer, value) for consistent seaborn handling.
    long_df = df.melt(
        id_vars=["qid", LABEL_COL],
        value_vars=list(SCORE_COLS.values()),
        var_name="ScoreKey",
        value_name="Value",
    )
    # Map flattened JSON keys back to human-facing scorer names (then shorten for axis labels).
    long_df["Score"] = long_df["ScoreKey"].map(COL_TO_NAME)
    long_df["Score"] = long_df["Score"].map(short_name)
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Value"])

    # Robust order: decide in long names, then map to short names for plotting.
    order_long = ordered_subset(DEFAULT_ORDER, list(SCORE_COLS.keys()))
    order = [short_name(s) for s in order_long]

    long_df["Score"] = pd.Categorical(long_df["Score"], categories=order, ordered=True)

    # Fixed string labels keep legend ordering stable across plots/exports.
    long_df["hallucinated_str"] = long_df[LABEL_COL].map({0: "No (0)", 1: "Yes (1)"})

    fig, ax = plt.subplots(figsize=(8.2, 4.5))

    # NOTE: potential issue: palette is hard-coded; keep consistent with other figures or Phase 2.
    pal = {"No (0)": "#4C72B0", "Yes (1)": "#DD8452"}

    sns.boxplot(
        data=long_df,
        x="Score",
        y="Value",
        hue="hallucinated_str",
        hue_order=["No (0)", "Yes (1)"],
        palette=pal,
        showfliers=True,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Score value")
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    
    ax.legend(
        title="Hallucinated",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
    )

    finalize_suptitle(fig, "Score Distributions by Hallucination Label", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "fig4_score_distributions.pdf")
    plt.close(fig)


def plot_figA_overall_distributions(df: pd.DataFrame, outdir: Path) -> None:
    """Appendix Figure A: overall score distributions (label-agnostic, same scorer ordering)."""
    long_df = df.melt(
        id_vars=[LABEL_COL],
        value_vars=list(SCORE_COLS.values()),
        var_name="ScoreKey",
        value_name="Value",
    )
    inv_map = {v: k for k, v in SCORE_COLS.items()}
    long_df["Scorer"] = long_df["ScoreKey"].map(inv_map)
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Value"])
    
    long_df["Scorer"] = long_df["Scorer"].map(short_name)

    order_long = ordered_subset(DEFAULT_ORDER, list(SCORE_COLS.keys()))
    order = [short_name(s) for s in order_long]
    long_df["Scorer"] = pd.Categorical(long_df["Scorer"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    sns.boxplot(data=long_df, x="Scorer", y="Value", color="lightgray", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Score value")
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    finalize_suptitle(fig, "Overall score distributions (all examples)", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "figA_overall_score_distributions.pdf")
    plt.close(fig)


def plot_figB_delta_auroc(manifest: dict, outdir: Path) -> None:
    """Appendix Figure B: ΔAUROC above random (AUROC − 0.5) with bootstrap CI."""
    auroc = manifest["scores"]["auroc"]
    boot = manifest["scores"]["bootstrap"]

    rows = []
    for label, k in MANIFEST_KEYS.items():
        if (k not in auroc) or (k not in boot):
            continue
        a = float(auroc[k])
        lo = float(boot[k]["ci_low"])
        hi = float(boot[k]["ci_high"])
        # Shift both point estimate and CI by the random baseline for direct "above chance" interpretation.
        rows.append({"Scorer": label, "delta": a - 0.5, "lo": lo - 0.5, "hi": hi - 0.5})

    b_df = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = np.arange(len(b_df))
    x = b_df["delta"].values
    xerr = np.vstack([x - b_df["lo"].values, b_df["hi"].values - x])

    ax.errorbar(x=x, y=y, xerr=xerr, fmt="o", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels([short_name(s) for s in b_df["Scorer"].tolist()])
    ax.axvline(0.0, linestyle="--", linewidth=1)  # delta=0 => random baseline
    ax.set_xlabel("ΔAUROC = AUROC − 0.5 (higher = better than random)")
    ax.set_ylabel("")
    ax.invert_yaxis()

    finalize_suptitle(fig, "AUROC margin above random (ΔAUROC) with bootstrap 95% CI", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "figB_delta_auroc_vs_random.pdf")
    plt.close(fig)


def plot_figC_score_overlap_density(df: pd.DataFrame, outdir: Path) -> None:
    """Appendix Figure C: label-conditional score overlap (KDE density; 2×2 panel grid)."""
    panel_order = [
        "LNTP (hallucination score)",
        "MTP (hallucination score)",
        "EGH probe (OOF)",
        "Hidden-state probe (OOF)",
    ]
    panel_order = ordered_subset(panel_order, list(SCORE_COLS.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharey=False)
    axes = axes.flatten()

    for ax, name in zip(axes, panel_order):
        col = SCORE_COLS[name]
        tmp = df[[LABEL_COL, col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])

        # NOTE: potential issue: KDE smoothing can be misleading for discrete/heavy-tailed scores; interpret qualitatively.
        sns.kdeplot(data=tmp[tmp[LABEL_COL] == 0], x=col, label="No (0)",  fill=True, alpha=0.35, ax=ax)
        sns.kdeplot(data=tmp[tmp[LABEL_COL] == 1], x=col, label="Yes (1)", fill=True, alpha=0.35, ax=ax)

        ax.set_title(short_name(name))
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")

    # Single shared legend to reduce visual clutter and keep label mapping consistent.
    axes[1].legend(title="Hallucinated", frameon=False)
    for ax in (axes[0], axes[2], axes[3]):
        ax.legend([], [], frameon=False)

    # Suptitle + layout exactly once (avoid double tight_layout calls that can shift panel geometry).
    fig.suptitle(
        "Score overlap by label (density plots)",
        y=0.995,
        fontsize=int(mpl.rcParams["figure.titlesize"] * 1.3),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    safe_savefig(fig, outdir / "figC_score_overlap_density.pdf")
    plt.close(fig)
    

def plot_figD_spearman_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    """Appendix Figure D: Spearman rank correlation across scorers (complete-case rows only)."""
    cols = {
        "LNTP": SCORE_COLS["LNTP (hallucination score)"],
        "MTP": SCORE_COLS["MTP (hallucination score)"],
        "EGH": SCORE_COLS["EGH probe (OOF)"],
        "Hidden": SCORE_COLS["Hidden-state probe (OOF)"],
    }

    # Complete-case filtering avoids pairwise-N differences that complicate interpretation.
    mat = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in cols.items()})
    mat = mat.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    corr = mat.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, square=True, cmap="coolwarm", ax=ax)

    finalize_suptitle(fig, "Spearman rank correlation between scorers", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "figD_spearman_correlation_heatmap.pdf")
    plt.close(fig)


def plot_figE_lntp_low_mismatches(df: pd.DataFrame, outdir: Path) -> None:
    """Appendix Figure E: inspect low-LNTP examples vs Hidden probe score (quantile threshold)."""
    x_col = SCORE_COLS["LNTP (hallucination score)"]
    y_col = SCORE_COLS["Hidden-state probe (OOF)"]

    tmp = df[[LABEL_COL, x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    # Heuristic: "LNTP-low" defined as bottom quartile on available (finite) LNTP scores.
    thr = float(tmp[x_col].quantile(0.25))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.scatterplot(data=tmp, x=x_col, y=y_col, hue=LABEL_COL, alpha=0.75, ax=ax)

    ax.axvline(thr, linestyle="--", linewidth=1)

    ax.text(
        thr + 0.01,
        tmp[y_col].quantile(0.55),
        f"LNTP low (q25 = {thr:.3f})",
        rotation=90,
        va="top",
        ha="left",
        fontsize=int(mpl.rcParams["xtick.labelsize"] * 0.8),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )

    ax.set_xlabel("LNTP score")
    ax.set_ylabel("Hidden score")
    # NOTE: potential issue: seaborn legend order depends on observed hue values; labels are forced for consistency.
    ax.legend(title="Hallucinated", labels=["No (0)", "Yes (1)"], frameon=False)

    finalize_suptitle(fig, "LNTP-low mismatches (low LNTP score vs Hidden-state probe)", y=0.97, top_rect=0.97)
    safe_savefig(fig, outdir / "figE_lntp_low_mismatches_vs_hidden.pdf")
    plt.close(fig)


# -----------------------------
# Optional extras (NOT paper)
# -----------------------------
def run_extras(df: pd.DataFrame) -> None:
    """
    Diagnostic-only exploratory analyses (excluded from the paper figure pipeline).
    NOTE: potential issue: results here depend on internal CV splits and are not part of the manifest-backed reporting.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print("\n[EXTRAS] Simulating OOF probe variants (diagnostic only).")

    # Example: probe with/without entropy
    X_old = df[["scores.egh_grad_norm", "scores.egh_emb_diff", "scores.egh_kl"]].to_numpy()
    X_new = df[["scores.egh_grad_norm", "scores.egh_emb_diff", "scores.egh_kl", "scores.egh_entropy"]].to_numpy()
    y = df[LABEL_COL].to_numpy()

    mask = np.isfinite(X_new).all(axis=1)
    X_old_clean, X_new_clean, y_clean = X_old[mask], X_new[mask], y[mask]

    def simulate_oof_probe(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> np.ndarray:
        """Train OOF logistic probe via stratified K-fold; returns per-example positive-class scores."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_scores = np.full(len(y), np.nan, dtype=float)
        for train_idx, test_idx in skf.split(X, y):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(random_state=random_state, max_iter=1000)),
                ]
            )
            pipe.fit(X[train_idx], y[train_idx])
            oof_scores[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
        return oof_scores

    oof_old = simulate_oof_probe(X_old_clean, y_clean)
    oof_new = simulate_oof_probe(X_new_clean, y_clean)

    auroc_old = roc_auc_score(y_clean, oof_old)
    auroc_new = roc_auc_score(y_clean, oof_new)

    print(f"Current (3 features):      AUROC = {auroc_old:.4f}")
    print(f"With entropy (4 features): AUROC = {auroc_new:.4f}")
    print(f"Improvement:               +{(auroc_new - auroc_old):.4f}")

    print("\n[EXTRAS] Simple ensemble probe weights (diagnostic only).")

    X = df[
        [
            "scores.lntp_uncertainty",
            "scores.mtp_uncertainty",
            "scores.egh_grad_norm",
            "scores.egh_emb_diff",
            "scores.hidden_probe_oof",
        ]
    ].to_numpy()
    y = df[LABEL_COL].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X_clean, y_clean = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, stratify=y_clean, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    names = ["LNTP", "MTP", "EGH_G", "EGH_E", "Hidden_probe"]
    print(f"Ensemble AUROC (held-out): {auc:.4f}")
    print("Learned weights:")
    for i, name in enumerate(names):
        print(f"  {name:12s}: {clf.coef_[0][i]: .4f}")


# -----------------------------
# CLI / main
# -----------------------------
@dataclass(frozen=True)
class Args:
    """Typed CLI configuration for reproducible, script-driven figure generation."""
    results: Path
    manifest: Path
    outdir: Path
    font_scale: float
    extras: bool


def parse_args() -> Args:
    """Parse CLI arguments and resolve paths to absolute locations."""
    p = argparse.ArgumentParser(
        description="Phase 1 Figures — TruthfulQA (Frozen Answers)"
    )

    # Base directory convention: project root is two levels above this script.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    default_results = project_root / "phase_1_replication" / "outputs" / "phase1_truthfulqa_hallu_results_300.jsonl"
    default_manifest = project_root / "phase_1_replication" / "outputs" / "phase1_run_manifest.json"
    default_outdir = project_root / "phase_1_replication" / "outputs" / "figs"

    p.add_argument("--results", type=str, default=str(default_results))
    p.add_argument("--manifest", type=str, default=str(default_manifest))
    p.add_argument("--outdir", type=str, default=str(default_outdir))
    p.add_argument("--font_scale", type=float, default=1.3)
    p.add_argument("--extras", action="store_true")

    a = p.parse_args()

    return Args(
        results=Path(a.results).resolve(),
        manifest=Path(a.manifest).resolve(),
        outdir=Path(a.outdir).resolve(),
        font_scale=float(a.font_scale),
        extras=bool(a.extras),
    )


def main() -> None:
    """Entry point: load inputs, print diagnostics, render all Phase 1 figures, and export PDFs."""
    args = parse_args()

    # Make seaborn respect matplotlib rcParams without imposing its own theme.
    sns.set_theme(style="white", rc={})

    # Apply Phase-2-like styling globally (single source of truth for typography/layout).
    apply_phase2_plot_style(font_scale=args.font_scale)

    ensure_dir(args.outdir)

    # Load data
    df = load_results_jsonl(args.results)
    manifest = load_manifest(args.manifest)

    # Diagnostic AUROC recomputation from JSONL (useful for catching score polarity/key mismatches).
    auroc_check = compute_auroc_from_results(df)
    print("\n[INFO] AUROC sanity check from results JSONL (not authoritative):")
    print(auroc_check.to_string(index=False))

    # Manifest AUROC/CI drives all reported figures (bootstrap details live in the manifest).
    ci_df = build_ci_table_from_manifest(manifest)
    print("\n[INFO] AUROC + bootstrap CI from manifest (authoritative for figures):")
    print(ci_df[["Scorer", "AUROC", "CI_low", "CI_high"]].to_string(index=False))

    # Produce figures
    plot_fig1_auroc_bar(ci_df, args.outdir)
    plot_fig2_bootstrap_ci(ci_df, args.outdir)
    plot_fig3_roc_overlay(df, args.outdir)
    plot_fig4_score_distributions(df, args.outdir)

    plot_figA_overall_distributions(df, args.outdir)
    plot_figB_delta_auroc(manifest, args.outdir)
    plot_figC_score_overlap_density(df, args.outdir)
    plot_figD_spearman_heatmap(df, args.outdir)
    plot_figE_lntp_low_mismatches(df, args.outdir)

    print(f"\n[OK] Saved figures to: {args.outdir.resolve()}")

    if args.extras:
        run_extras(df)


if __name__ == "__main__":
    main()