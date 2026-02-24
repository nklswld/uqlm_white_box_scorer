"""
Analyze ablations of EGH probe components (G-only, E-only, G+E) from finalized runs.

Inputs: per-run *.results.jsonl (example-level labels + score fields) and
        *.manifest.bootstrap_indices.npz (precomputed resample indices),
        discovered under outputs/final/ via matching *.manifest.json prefixes.
Outputs: CSV of per-run bootstrap summary metrics and publication-ready overlay + matrix figures.
Reproducibility: fully deterministic given fixed bootstrap index files and unchanged finalized artifacts.
"""

# phase_2_medical/analysis/ablations/analyze_egh_ge_components.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: consistent with phase2_figures.py
# ============================================================
# Global Matplotlib style for cross-figure comparability (font sizing, tick weights, PDF text embedding).
FONT_SCALE = 1.5

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

    "font.size": int(12 * FONT_SCALE),
    "axes.titlesize": int(14 * FONT_SCALE),
    "axes.labelsize": int(13 * FONT_SCALE),
    "xtick.labelsize": int(13 * FONT_SCALE),
    "ytick.labelsize": int(10 * FONT_SCALE),
    "legend.fontsize": int(11 * FONT_SCALE),
    "legend.title_fontsize": int(11 * FONT_SCALE),
    "figure.titlesize": int(15 * FONT_SCALE),

    "axes.titlepad": 12,

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,

    # Ensure text remains editable in vector exports (avoid Type 3 fonts).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

# Fixed y-limits improve between-panel comparability; update only when the expected metric range expands.
AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.05, 0.60)

ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}


# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a figure; if the target file is locked (common on Windows), write a versioned fallback instead."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        stem, suffix = outpath.stem, outpath.suffix
        # Retry with suffix-versioning to preserve an otherwise successful run when a PDF is open in a viewer.
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
    """Annotate point estimates above the CI upper whisker; skips NaN/None values."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span  # padding scales with axis span to remain readable across different y-limits

    for x, y, eh in zip(x_positions, y_values, yerr_high):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        top = float(y) + (0.0 if eh is None else float(eh))
        ax.text(float(x), top + pad, fmt.format(float(y)),
                ha="center", va="bottom", fontsize=fontsize)


# ============================================================
# IO helpers
# ============================================================
def load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts (one dict per non-empty line)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def np_load_first_array(npz_path: Path):
    """Load a bootstrap index array from NPZ: prefer common keys; otherwise return the first stored array."""
    z = np.load(npz_path, allow_pickle=True)
    # Convention: accept multiple historical key names to avoid coupling to a single producer script/version.
    for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if k in z.files:
            return z[k]
    return z[z.files[0]]

def load_bootstrap_indices(boot_path: Path):
    """Load bootstrap index matrix (B, N) from run artifacts.

    Prefer the EGH-GE index field when present so all EGH-component comparisons share identical resamples.
    """
    z = np.load(boot_path, allow_pickle=True)

    # Prefer stable, explicitly named arrays; fall back to older single-array conventions.
    for k in ["egh_ge", "egh", "indices", "boot_idx", "bootstrap_indices", "idx"]:
        if k in z.files:
            arr = z[k]
            break
    else:
        arr = z[z.files[0]]

    # Some producers store each bootstrap draw as an object-array row; normalize to a dense (B, N) int array.
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)

def find_label_key(example: dict):
    """Select the label key from a result row (first match among known conventions)."""
    # NOTE: potential issue: this relies on key presence only; mixed schemas across files can mislabel silently.
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    """Return a lowercased mapping of score-name -> float score, supporting multiple result schemas."""
    # Primary schema: nested dict of named scores.
    if "scores" in example and isinstance(example["scores"], dict):
        return {str(k).lower(): float(v) for k, v in example["scores"].items()}
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return {str(k).lower(): float(v) for k, v in example["wb_scores"].items()}

    # Fallback schema: treat all numeric top-level fields as candidate scores.
    # NOTE: potential issue: numeric metadata may be swept in; downstream filtering must remain strict.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            scores[str(k).lower()] = float(v)
    return scores

def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and choose score polarity so 'higher score => positive class' yields AUROC >= 0.5."""
    # Convention: per run/category, flip score sign if AUROC < 0.5 so reported AUROC is always >= 0.5.
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0

def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and equal-tailed CI for AUROC using precomputed indices; skip degenerate resamples."""
    # Determinism: resampling is entirely determined by boot_idx loaded from disk (no RNG used here).
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Silent failure mode: AUROC is undefined without class variation; these draws are dropped.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))

    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        # All resamples were degenerate (or input invalid); propagate NaNs to plotting/export.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi

def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and equal-tailed CI for Spearman ρ using precomputed indices; skip degenerate/NaN draws."""
    # NOTE: potential issue: Spearman on binary y reflects monotonic association; interpret as rank-based effect proxy.
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Silent failure mode: correlation is undefined/uninformative without label variation; these draws are dropped.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        if pd.isna(rho):
            continue
        rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi


# ============================================================
# Paths (scan outputs/final/)
# ============================================================
# Project root inference: this file is expected under phase_2_medical/analysis/ablations/.
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
FINAL_ROOT = ROOT / "outputs" / "final"
OUT_DIR = ROOT / "outputs" / "figures_tables" / "ablations" / "egh_components"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT =", ROOT)
print("FINAL_ROOT =", FINAL_ROOT)
print("FINAL_ROOT exists =", FINAL_ROOT.exists())
print("OUT_DIR =", OUT_DIR)

if not FINAL_ROOT.exists():
    raise FileNotFoundError(f"Final folder not found: {FINAL_ROOT}")


# ============================================================
# Discover runs in final
# ============================================================
def parse_task_model_from_prefix(prefix: str):
    """Parse (task, model) from a run filename prefix; ignore any post-dot tags and normalize common model families."""
    base = prefix.split(".")[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse task/model from prefix: {prefix}")

    task = parts[0].lower()
    model_raw = "_".join(parts[1:]).lower()

    # Heuristic normalization keeps legends consistent across runs with different naming granularity.
    if "bio" in model_raw:
        model = "biomistral"
    elif "mistral" in model_raw:
        model = "mistral"
    else:
        model = model_raw

    return task, model

runs = []
for manifest_path in sorted(FINAL_ROOT.glob("*.manifest.json")):
    prefix = manifest_path.name.replace(".manifest.json", "")
    results_path = manifest_path.with_name(prefix + ".results.jsonl")
    boot_path = manifest_path.with_name(prefix + ".manifest.bootstrap_indices.npz")

    # Missing artifacts imply the run is not evaluable for CI; skip but warn for auditability.
    if not results_path.exists():
        print("[WARN] Missing results:", results_path)
        continue
    if not boot_path.exists():
        print("[WARN] Missing bootstrap:", boot_path)
        continue

    task, model = parse_task_model_from_prefix(prefix)
    runs.append((task, model, manifest_path, results_path, boot_path))

print("Found final runs:", len(runs))
if len(runs) == 0:
    listing = sorted([p.name for p in FINAL_ROOT.glob("*")][:200])
    raise RuntimeError("No final runs found. Sample listing:\n" + "\n".join(listing))


# ============================================================
# EGH component keys (from your results.jsonl)
# ============================================================
# These keys must exist as score fields for every evaluated run; otherwise the run is excluded.
CATS = ["egh_probe_g_only", "egh_probe_e_only", "egh_probe_ge"]
CAT_PRETTY = {
    "egh_probe_g_only": "G-only",
    "egh_probe_e_only": "E-only",
    "egh_probe_ge": "G+E",
}


# ============================================================
# Compute metrics per run (AUROC + Spearman with CI)
# ============================================================
records = []
for task, model, manifest_path, results_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        continue

    # Label extraction: binary y is cast to int; positive class is "1" by roc_auc_score convention.
    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # Score extraction: intersect keys across rows to preserve strict example-level alignment (no implicit NaNs).
    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}

    # Hard requirement: all EGH ablation categories must be present; otherwise comparisons are not like-for-like.
    missing = [k for k in CATS if k not in S]
    if missing:
        print(f"[WARN] {results_path.name}: missing keys {missing}; skipping this run.")
        continue

    boot_idx = load_bootstrap_indices(boot_path)

    for cat in CATS:
        s_raw = S[cat]
        # Polarity normalization is applied before bootstrapping so CIs correspond to the reported direction.
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "category": cat,
            "direction": float(direction),
            "N": int(len(y)),
            "pos_rate": float(y.mean()),

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
out_csv = OUT_DIR / "analysis_egh_components_metrics.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

if df.empty:
    raise RuntimeError("No rows computed for EGH components. Check final results contain the expected keys.")


# ============================================================
# Plotting: Overlay per task (lines=model), + Bars per task×model
# Includes: reindexing to full CATS (Fix), shaded CI band + black whiskers (Fix)
# ============================================================
# Order tasks/models to match paper conventions when available; append any additional discovered labels.
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])


def plot_overlay(metric: str, y_lim, title: str, outpath: Path):
    """Plot per-task overlays with one line per model and 95% CI shading/whiskers."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    x = np.arange(len(CATS), dtype=float)

    for ax, task in zip(axes, tasks):
        for model in models:
            sub = df[(df["task"] == task) & (df["model"] == model)].copy()
            if sub.empty:
                continue

            # Invariant: x positions always correspond to CATS order (even if a row is missing/NaN post-filtering).
            sub = sub.set_index("category").reindex(CATS).reset_index()

            if metric == "auroc":
                y = sub["auroc_boot_mean"].to_numpy(dtype=float)
                lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                ylabel = "AUROC"
                hline = 0.5  # random-guess baseline for AUROC
            else:
                y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                ylabel = "Spearman ρ (bootstrap mean)"
                hline = 0.0  # null association baseline for correlation

            # CI shading only where all bounds are finite (avoids warnings / misleading bands over NaNs).
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.15)

            ax.plot(x, y, marker="o", label=MODEL_PRETTY.get(model, model))
            yerr = np.vstack([y - lo, hi - y])
            # NOTE: potential issue: errorbars are computed for all x; NaNs may be ignored or warn depending on backend.
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=ERRORBAR_CAPSIZE,
                        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK, ecolor="black")

        ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
        ax.set_xticks(x)
        ax.set_xticklabels([CAT_PRETTY.get(c, c) for c in CATS], rotation=0, ha="center")
        ax.margins(x=0.12)
        ax.set_ylim(*y_lim)
        overlay_panel_title_fs = mpl.rcParams["axes.titlesize"] * 0.90
        ax.set_title(TASK_PRETTY.get(task, task), fontsize=overlay_panel_title_fs)
        ax.set_ylabel(ylabel)

    axes[0].legend(frameon=False, title="Model")
    axes[0].legend(frameon=False, title="Model")

    fig.suptitle(title, y=0.98, fontsize=mpl.rcParams["figure.titlesize"])
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.subplots_adjust(wspace=0.20)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)
    plt.close(fig)
    print("Wrote:", outpath)

def plot_bars_matrix(metric: str, y_lim, outstem: str):
    """Create a 2x2 barplot matrix per metric (rows=tasks, cols=models) with consistent axes and 95% CI whiskers."""
    # Enforce requested layout order (if present) so figure panels align with manuscript narrative.
    tasks_order = [t for t in ["medqa", "pubmedqa"] if t in tasks]
    models_order = [m for m in ["mistral", "biomistral"] if m in models]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.2, 9.2),
        sharey=True
    )

    x = np.arange(len(CATS), dtype=float)

    for r, task in enumerate(tasks_order):
        for c, model in enumerate(models_order):
            ax = axes[r, c]
            sub = df[(df["task"] == task) & (df["model"] == model)].copy()

            # Keep panel empty but structurally consistent (fixed axes/labels) when a task×model cell is absent.
            if sub.empty:
                ax.set_xticks(x)
                ax.set_xticklabels([CAT_PRETTY.get(cat, cat) for cat in CATS])
                ax.set_ylim(*y_lim)
                ax.set_title(f"{TASK_PRETTY.get(task, task)} — {MODEL_PRETTY.get(model, model)}")
                ax.grid(False)
                continue

            # Invariant: bars are always ordered as CATS for direct within-row comparisons.
            sub = sub.set_index("category").reindex(CATS).reset_index()

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
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=ERRORBAR_CAPSIZE,
                        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK, ecolor="black")
            ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)

            ax.set_xticks(x)
            ax.set_xticklabels([CAT_PRETTY.get(cat, cat) for cat in CATS])
            ax.set_ylim(*y_lim)

            # Panel title encodes (task, model) for unambiguous caption/appendix references.
            ax.set_title(f"{TASK_PRETTY.get(task, task)} — {MODEL_PRETTY.get(model, model)}")

            # Only left column gets y-label to reduce redundant ink while preserving readability.
            if c == 0:
                ax.set_ylabel(ylabel)

            add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

    fig.suptitle(
        f"EGH Probe Design (G-only vs E-only vs G+E) — "
        f"{'AUROC' if metric == 'auroc' else 'Spearman ρ'} ± 95% CI",
        y=0.955,
        fontsize=mpl.rcParams["figure.titlesize"] * 1.1,
    )

    # Layout: reserve headroom for the suptitle and avoid inter-panel label overlaps.
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.35, wspace=0.11)

    safe_savefig(fig, OUT_DIR / f"{outstem}_{metric}_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", OUT_DIR / f"{outstem}_{metric}_matrix.pdf")

plot_overlay(
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="EGH Probe Design (G-only vs E-only vs G+E) — AUROC ± 95% CI",
    outpath=OUT_DIR / "fig_ablation_egh_components_auroc_overlay.pdf",
)
plot_overlay(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="EGH Probe Design (G-only vs E-only vs G+E) — Spearman ρ ± 95% CI",
    outpath=OUT_DIR / "fig_ablation_egh_components_spearman_overlay.pdf",
)

plot_bars_matrix(metric="auroc", y_lim=AUROC_YLIM, outstem="fig_ablation_egh_components")
plot_bars_matrix(metric="spearman", y_lim=SPEARMAN_YLIM, outstem="fig_ablation_egh_components")

print("[OK] EGH components ablation done. Outputs in:", OUT_DIR)