"""
Analyze hidden-pooling ablation runs and produce publication-ready figures/tables.

Inputs: per-run *.results.jsonl (labels + score fields) and *.manifest.bootstrap_indices.npz
(bootstrap resample indices, optionally per-score and hidden-kept indices for subset alignment).
Outputs: a CSV with per-run metrics (AUROC/Spearman with 95% bootstrap CI) and PDF plots
(overlay + 2x2 bar matrices) saved under outputs/figures_tables/ablations/hidden_pooling/.
Determinism: statistics are deterministic given fixed bootstrap indices stored in the NPZ files.
"""

# phase_2_medical/analysis/ablations/analyze_hidden_pooling.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: EXACTLY consistent with phase2_figures.py
# ============================================================
FONT_SCALE = 1.5  # Global typography scale (must match phase2_figures.py for visual consistency)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

    "font.size": int(12 * FONT_SCALE),
    "axes.titlesize": int(13 * FONT_SCALE),
    "axes.labelsize": int(12 * FONT_SCALE),
    "xtick.labelsize": int(13 * FONT_SCALE),
    "ytick.labelsize": int(11 * FONT_SCALE),
    "legend.fontsize": int(11 * FONT_SCALE),
    "legend.title_fontsize": int(11 * FONT_SCALE),
    "figure.titlesize": int(16 * FONT_SCALE),

    "axes.titlepad": 12,

    # Slightly thicker axes/ticks for print/PDF legibility
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

# Global y-limits enforce comparability across panels/runs (avoid per-plot autoscale bias).
AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.05, 0.60)

# ---------------------------------------------------------------------
# Plot styling knobs (global, for print/readability)
# ---------------------------------------------------------------------
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

# Human-readable labels (keep stable across figures/tables).
TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}
def pretty_score(k: str) -> str:
    """Return a display label for a score key (fallback: lowercase key)."""
    kk = str(k).lower()
    return SCORE_PRETTY.get(kk, kk)


# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a figure, retrying with versioned filenames if the target PDF is locked/open."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        # NOTE: potential issue: repeated PermissionError typically means the PDF is open in a viewer.
        stem, suffix = outpath.stem, outpath.suffix
        for k in range(2, 50):
            alt = outpath.with_name(f"{stem}_v{k}{suffix}")
            try:
                fig.savefig(alt, **kwargs)
                print(f"[WARN] Permission denied for {outpath.name} (likely open). Wrote: {alt.name}")
                return alt
            except PermissionError:
                continue
        raise


# ============================================================
# Helper: labels above CI
# ============================================================
def add_value_labels_above_ci(ax, x_positions, y_values, yerr_high,
                              fmt="{:.3f}", fontsize=None, pad_frac=0.015):
    """Annotate points/bars with their mean value, placed just above the CI upper whisker."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    # Use axes span so label padding scales with fixed y-limits (consistent across panels).
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span

    for x, y, eh in zip(x_positions, y_values, yerr_high):
        # Skip missing points to avoid misleading annotations.
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        top = float(y) + (0.0 if eh is None else float(eh))
        ax.text(float(x), top + pad, fmt.format(float(y)),
                ha="center", va="bottom", fontsize=fontsize)


# ============================================================
# IO helpers
# ============================================================
def load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts (skips empty lines)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def boot_key_for_score(score_key: str) -> str | None:
    """
    Map a score key to the corresponding bootstrap-index array name stored in the NPZ.

    Returns None to indicate "legacy" convention (single bootstrap array without per-score keys).
    """
    k = (score_key or "").lower()

    # Hidden probe
    if "hidden" in k:
        return "hidden"

    # LNTP / MTP
    if k == "lntp":
        return "lntp"
    if k == "mtp":
        return "mtp"

    # EGH family
    if k.startswith("egh_probe_ge"):
        return "egh_ge"
    if k.startswith("egh_probe_g_only"):
        return "egh_g"
    if k.startswith("egh_probe_e_only"):
        return "egh_e"
    if k.startswith("egh_probe_scalar_only"):
        return "egh_scalar"
    if "egh" in k:
        # Legacy/default in run_phase2 npz often has `egh` too.
        return "egh"

    return None


def load_bootstrap_indices(boot_path: Path, score_key: str | None = None) -> np.ndarray:
    """
    Load bootstrap resampling indices with shape (B, N) as int.

    Selection rule: prefer per-score arrays (e.g., lntp/mtp/egh_*/hidden), else fall back to
    legacy single-array keys; final fallback uses the first NPZ entry.
    """
    z = np.load(boot_path, allow_pickle=True)

    preferred = None
    if score_key is not None:
        # Heuristic mapping ensures we pick the correct bootstrap stream for the metric.
        bk = boot_key_for_score(score_key)
        if bk is not None and bk in z.files:
            preferred = z[bk]

    if preferred is None:
        # Legacy single-array conventions.
        for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
            if k in z.files:
                preferred = z[k]
                break

    if preferred is None:
        # NOTE: potential issue: "first entry" fallback can silently change behavior if NPZ layout changes.
        preferred = z[z.files[0]]

    arr = preferred
    # Some writers store boot indices as an object array of arrays; stack into (B, N).
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)

    return arr.astype(int)


def load_hidden_kept_indices(boot_path: Path) -> np.ndarray | None:
    """Return the kept-example indices for hidden probe alignment if stored in the NPZ; else None."""
    z = np.load(boot_path, allow_pickle=True)
    if "hidden_kept_indices" in z.files:
        return z["hidden_kept_indices"].astype(int)
    return None

def find_label_key(example: dict):
    """Infer the binary label field name from common conventions used across result JSONL writers."""
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    """Extract per-example score dict, supporting multiple writer conventions and numeric fallbacks."""
    # Canonical nested dictionaries.
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    # Fallback: pick numeric fields that look like score keys.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)
    return scores

def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and (if needed) flip score sign so AUROC >= 0.5; returns (auroc, direction)."""
    # Convention: direction=+1 means "as-is", direction=-1 means score is negated for interpretability.
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0

def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and central (1-alpha) CI for AUROC using precomputed resampling indices."""
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Skip degenerate resamples with no label variation (roc_auc_score would error / be undefined).
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))

    aucs = np.asarray(aucs, dtype=float)

    # handle pathological cases where ALL bootstrap samples were invalid
    if aucs.size == 0:
        # NOTE: potential issue: downstream plots will show NaNs; treat as "insufficient variation" not "0".
        return np.nan, np.nan, np.nan

    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi

def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and central (1-alpha) CI for Spearman correlation using resampling indices."""
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Skip degenerate resamples (constant labels => undefined rank correlation).
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        # Pandas may return NaN for pathological inputs; skip to keep CI well-defined.
        if pd.isna(rho):
            continue
        rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)

    # handle pathological cases where ALL bootstrap samples were invalid
    if rhos.size == 0:
        # NOTE: potential issue: often indicates extreme class imbalance or filtering reduced N too far.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi


# ============================================================
# Paths (scan outputs/ablations/hidden_pooling/)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
ABL_ROOT = ROOT / "outputs" / "ablations" / "hidden_pooling"
OUT_DIR = ROOT / "outputs" / "figures_tables" / "ablations" / "hidden_pooling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal diagnostics to confirm run discovery and output locations.
print("ROOT =", ROOT)
print("ABL_ROOT =", ABL_ROOT)
print("ABL_ROOT exists =", ABL_ROOT.exists())
print("OUT_DIR =", OUT_DIR)

if not ABL_ROOT.exists():
    raise FileNotFoundError(f"Hidden pooling folder not found: {ABL_ROOT}")


# ============================================================
# Discover runs
# Expect structure: hidden_pooling/<task>_<model>/<pooling>/*.manifest.json
# and same directory contains:
#   <prefix>.results.jsonl
#   <prefix>.manifest.bootstrap_indices.npz
# where <prefix> is manifest filename without ".manifest.json"
# ============================================================
runs = []
for manifest_path in sorted(ABL_ROOT.glob("**/*.manifest.json")):
    # Convention: folder names encode experimental factors (task/model) and pooling strategy.
    # NOTE: potential issue: parsing assumes "<task>_<model...>" with "_" as separator.
    try:
        pooling = manifest_path.parent.name
        task_model = manifest_path.parent.parent.name

        parts = task_model.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected folder format: {task_model}")

        task = parts[0].lower()
        model_raw = "_".join(parts[1:]).lower()

        # Normalize model naming (robust against e.g. bio_mistral_large etc.).
        if "bio" in model_raw:
            model = "biomistral"
        elif "mistral" in model_raw:
            model = "mistral"
        else:
            model = model_raw

    except Exception:
        print("[WARN] Could not parse folders for:", manifest_path)
        continue

    prefix = manifest_path.name.replace(".manifest.json", "")
    results_path = manifest_path.with_name(prefix + ".results.jsonl")
    boot_path = manifest_path.with_name(prefix + ".manifest.bootstrap_indices.npz")

    if not results_path.exists():
        print("[WARN] Missing results:", results_path)
        continue
    if not boot_path.exists():
        print("[WARN] Missing bootstrap:", boot_path)
        continue

    runs.append((task, model, pooling, manifest_path, results_path, boot_path))

print("Found runs:", len(runs))
if len(runs) == 0:
    listing = sorted([str(p.relative_to(ABL_ROOT)) for p in ABL_ROOT.glob("**/*")][:200])
    raise RuntimeError("No runs found. Sample listing:\n" + "\n".join(listing))


# ============================================================
# Compute metrics per run (AUROC + Spearman with CI)
# (Fix 3 NOT applied: keep all available score keys)
# ============================================================
records = []
for task, model, pooling, manifest_path, results_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        continue

    # Label extraction is convention-based; label must be binary (0/1) for AUROC.
    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)
  
    # Normalize score keys to lowercase for stable matching across writers.
    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]

    # Invariant: only evaluate score keys present for every example (prevents silent length mismatch).
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted(keys)

    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
    

    for score_key, s_raw in S.items():
        # Reproducibility: resampling is driven by stored indices, not an RNG at runtime.
        boot_idx = load_bootstrap_indices(boot_path, score_key=score_key)

        # --- NEW: align to kept subset for hidden probe ---
        # Hidden probe may be computed on a filtered subset; align y/s and bootstrap N accordingly.
        y_use = y
        s_use = s_raw
        if score_key.lower() == "hidden_probe_oof":
            kept = load_hidden_kept_indices(boot_path)
            if kept is not None:
                y_use = y[kept]
                s_use = s_raw[kept]
            else:
                # fallback: drop NaN/inf consistently if kept indices not available
                # NOTE: this finite-mask fallback assumes it matches the original hidden-probe filtering logic; 
                # validate if results are sensitive to the kept-set definition.
                m = np.isfinite(s_raw)
                y_use = y[m]
                s_use = s_raw[m]

        # Optional hard guard: catches silent shape mismatches immediately
        if boot_idx.shape[1] != len(y_use):
            raise ValueError(
                f"Bootstrap shape mismatch for {score_key}: "
                f"boot_idx {boot_idx.shape} vs N={len(y_use)} (file={boot_path.name})"
            )

        # AUROC direction: choose sign so "higher score => higher error probability" is consistent.
        au, direction = auroc_with_best_direction(y_use, s_use)
        s = s_use * direction

        # Bootstrap uses the direction-aligned score vector to keep CI consistent with reported AUROC.
        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y_use, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y_use, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "pooling": pooling,
            "score_key": score_key,
            "direction": float(direction),
            "N": int(len(y_use)),
            "pos_rate": float(y_use.mean()) if len(y_use) > 0 else np.nan,

            "auroc": float(au),
            "auroc_boot_mean": float(au_mean),
            "auroc_ci95_lo": float(au_lo),
            "auroc_ci95_hi": float(au_hi),

            "spearman_rho_boot_mean": float(sp_mean),
            "spearman_ci95_lo": float(sp_lo),
            "spearman_ci95_hi": float(sp_hi),

            "manifest_file": str(manifest_path),
            "results_file": str(results_path),
            "boot_file": str(boot_path),
        })

df = pd.DataFrame(records)
out_csv = OUT_DIR / "analysis_hidden_pooling_metrics.csv"  # keep as you want (Fix 2 NOT applied)
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)


# ============================================================
# Plotting
# We focus primarily on Hidden probe for this ablation
# ============================================================
df_hid = df[df["score_key"].str.lower().eq("hidden_probe_oof")].copy()
if df_hid.empty:
    raise KeyError(
        "hidden_probe_oof not found in this ablation run. "
        "Ablation plots are defined for this score only."
    )

# Stable categorical order improves across-run comparability and prevents accidental resorting.
POOL_ORDER_HINT = ["last_answer", "mean_answer", "mean_all"]
poolings = [p for p in POOL_ORDER_HINT if p in set(df_hid["pooling"])]
poolings += sorted([p for p in set(df_hid["pooling"]) if p not in poolings])

tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df_hid["task"])]
tasks += sorted([t for t in set(df_hid["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df_hid["model"])]
models += sorted([m for m in set(df_hid["model"]) if m not in models])


# ============================================================
# Plot overlay with CI band + black whiskers (Fix 4 + Fix 5 + C)
# - 1 panel per task
# - lines = models
# - x = pooling (categorical)
# ============================================================
def plot_overlay(metric: str, y_lim, title: str, outpath: Path):
    """Plot per-task overlays (lines=models) with bootstrap CI bands/whiskers for a given metric."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        x = np.arange(len(poolings), dtype=float)

        for model in models:
            sub = df_hid[(df_hid["task"] == task) & (df_hid["model"] == model)].copy()
            if sub.empty:
                continue

            # Fix 4: reindex to full pooling list to enforce alignment across models (even if some are missing).
            sub = sub.set_index("pooling").reindex(poolings).reset_index()

            if metric == "auroc":
                y = sub["auroc_boot_mean"].to_numpy(dtype=float)
                lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                ylabel = "AUROC"
                hline = 0.5
            else:
                y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                ylabel = "Spearman ρ (bootstrap mean)"
                hline = 0.0

            # Fix 5: CI band (with NaN-safe masking to avoid matplotlib warnings/shape surprises).
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.15)

            # Line plot conveys trend across pooling strategies; markers aid black/white print readability.
            ax.plot(x, y, marker="o", label=MODEL_PRETTY.get(model, model))

            # Black whiskers (phase2-like): CI shown as error bars even when band is present.
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
                        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)

            # Annotate best pooling per model (NaN-safe).
            if np.isfinite(y).any():
                best_i = int(np.nanargmax(y))

                # Slight horizontal offset reduces overlap when multiple models share the same best pooling.
                dx = 0.02
                if best_i in [0, 1]:   # last_answer, mean_answer
                    x_text = x[best_i] + dx
                    ha = "left"
                else:                  # mean_all
                    x_text = x[best_i] - dx
                    ha = "right"

                ax.text(
                    x_text,
                    y[best_i] + 0.01 * (y_lim[1] - y_lim[0]),
                    f"{y[best_i]:.3f}",
                    ha=ha,
                    va="bottom",
                    fontsize=VALUE_LABEL_FONTSIZE,
                )

        # Baseline provides a null reference (AUROC=0.5, Spearman=0.0).
        ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
        ax.set_xticks(np.arange(len(poolings)))
        ax.set_xticklabels(poolings, rotation=20, ha="right")
        ax.set_ylim(*y_lim)
        ax.set_title(TASK_PRETTY.get(task, task))
        ax.set_ylabel(ylabel)

    axes[0].legend(frameon=False, title="Model")
    fig.suptitle(title, y=1.04, fontsize=plt.rcParams["figure.titlesize"])
    fig.subplots_adjust(wspace=0.24, top=0.82)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


def plot_bars_matrix(metric: str, y_lim, outpath: Path):
    """
    2x2 matrix for fixed task/model layout; bars show pooling strategies with 95% CI whiskers.

    Layout: cols = [mistral, biomistral], rows = [medqa, pubmedqa]; x = pooling (categorical).
    """
    tasks_order = ["medqa", "pubmedqa"]
    models_order = ["mistral", "biomistral"]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.2, 8.6),
        sharey=True
    )
    axes = np.array(axes)

    x = np.arange(len(poolings), dtype=float)

    for r, task in enumerate(tasks_order):
        for c, model in enumerate(models_order):
            ax = axes[r, c]

            sub = df_hid[(df_hid["task"] == task) & (df_hid["model"] == model)].copy()
            if sub.empty:
                # Keep grid shape stable; absent combinations are intentionally blank.
                ax.set_axis_off()
                continue

            # Reindex enforces identical category order and bar positions across all panels.
            sub = sub.set_index("pooling").reindex(poolings).reset_index()

            if metric == "auroc":
                y = sub["auroc_boot_mean"].to_numpy(dtype=float)
                lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                ylabel = "AUROC"
                hline = 0.5
            else:
                y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                ylabel = "Spearman ρ (bootstrap mean)"
                hline = 0.0

            yerr_low = y - lo
            yerr_high = hi - y

            ax.bar(x, y, width=0.65)
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black", 
                elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)
            ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)

            ax.set_xticks(x)
            ax.set_xticklabels(poolings, rotation=20, ha="right")

            # Global y-limits: ensures visual comparability across the full 2x2 grid.
            ax.set_ylim(*y_lim)

            ax.set_title(f"{TASK_PRETTY.get(task, task)} — {MODEL_PRETTY.get(model, model)}")

            if c == 0:
                ax.set_ylabel(ylabel)

            # Place value labels above CI to avoid overlapping whiskers (NaN-safe inside helper).
            add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

    fig.suptitle(
        "Hidden Probe Pooling — "
        + ("AUROC ± 95% CI" if metric == "auroc" else "Spearman ρ ± 95% CI"),
        y=0.97,
        fontsize=plt.rcParams["figure.titlesize"],
    )

    # Increased vertical spacing improves readability when axis labels are rotated.
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.subplots_adjust(hspace=0.65, wspace=0.14)

    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)
    

# Overlay figures (story-like; per task; lines = models; with CI band)
plot_overlay(
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="Hidden Probe Pooling — AUROC ± 95% CI (Hidden; by task; lines = models)",
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_auroc_overlay.pdf",
)
plot_overlay(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="Hidden Probe Pooling — Spearman ρ ± 95% CI (Hidden; by task; lines = models)",
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_spearman_overlay.pdf",
)

# 2x2 matrices (AUROC + Spearman)
plot_bars_matrix(
    metric="auroc",
    y_lim=AUROC_YLIM,
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_auroc_matrix.pdf",
)
plot_bars_matrix(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_spearman_matrix.pdf",
)

print("[OK] Hidden pooling ablation done. Outputs in:", OUT_DIR)