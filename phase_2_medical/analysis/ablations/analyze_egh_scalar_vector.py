"""
Analyze EGH probe ablations by comparing a scalar-only probe against a vector (G+E) probe.
Inputs are per-run JSONL result files (labels + score fields) and corresponding NPZ bootstrap indices,
auto-discovered under outputs/final via *.manifest.json naming conventions.
Outputs are (i) a CSV of per-run bootstrap means and 95% CIs for AUROC and Spearman rho, and (ii) PDF figures.
Determinism: results are deterministic given fixed input files and stored bootstrap indices (no RNG used here).
"""

# phase_2_medical/analysis/ablations/analyze_egh_scalar_vs_vector.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Plot style (kept consistent with phase2_figures.py for cross-figure comparability)
# ============================================================
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
    
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

# Global y-limits enforce consistent scaling across tasks/models within a metric panel.
AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.05, 0.60)

ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}


# ============================================================
# Robust save helper (common failure: PDF open in viewer on Windows)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a figure, falling back to versioned filenames if the target path is locked."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
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
    """Annotate plotted points/bars with y-values, placed above the upper CI whisker."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    # Pad in axis units to avoid label/whisker collisions across different y-limits.
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span

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
    """Load a JSONL file into a list of dicts (skipping empty lines)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_bootstrap_indices(boot_path: Path, key: str | None = None):
    """
    Load bootstrap resample indices as a dense int array of shape [B, N].

    If `key` is provided, prefer that array from the NPZ (phase-2 saves multiple
    index matrices per metric, e.g. egh_ge, egh_scalar, ...). Falls back to a
    reasonable default if the key is missing.
    """
    z = np.load(boot_path, allow_pickle=True)

    # Prefer explicit key when available (most important for phase-2 artifacts)
    if key is not None and key in z.files:
        arr = z[key]
    else:
        # Backward-compat / older artifacts: try common generic names
        for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
            if k in z.files:
                arr = z[k]
                break
        else:
            # Last resort: first stored array (kept for robustness, but not ideal)
            arr = z[z.files[0]]

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)

def find_label_key(example: dict):
    """Infer the label field name from a single example row using a fixed priority order."""
    # Convention: prefer explicit error/label keys; fall back to common supervised-learning names.
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    """Extract a {score_name -> float} mapping, normalizing keys to lowercase."""
    # Preferred: a nested scores dict (more explicit + avoids accidentally collecting metadata floats).
    if "scores" in example and isinstance(example["scores"], dict):
        return {str(k).lower(): float(v) for k, v in example["scores"].items()}
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return {str(k).lower(): float(v) for k, v in example["wb_scores"].items()}

    # Fallback: treat any numeric top-level fields as candidate scores.
    # NOTE: potential issue: this can inadvertently include non-score numeric metadata if present.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            scores[str(k).lower()] = float(v)
    return scores

def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and flip score polarity if needed so higher score implies higher positive class."""
    # Polarity convention: enforce AUROC >= 0.5 by optionally negating scores.
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0

def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and two-sided (1-alpha) CI for AUROC using precomputed resample indices."""
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Silent skip: degenerate resample with a single class makes AUROC undefined.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))

    aucs = np.asarray(aucs, dtype=float)
    # NOTE: potential issue: if many resamples are degenerate, CI may be unstable (few effective draws).
    if aucs.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi

def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and two-sided (1-alpha) CI for Spearman rho using precomputed resample indices."""
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Keep consistency with AUROC path: skip degenerate-label resamples.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        # Silent skip: correlation can be NaN for constant/ill-conditioned samples.
        if pd.isna(rho):
            continue
        rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)
    # NOTE: potential issue: effective bootstrap sample size can shrink after NaN/degeneracy filtering.
    if rhos.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi


# ============================================================
# Paths (scan outputs/final/)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
FINAL_ROOT = ROOT / "outputs" / "final"
OUT_DIR = ROOT / "outputs" / "figs" / "ablations" / "egh_scalar_vs_vector"
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
    """Parse task/model from filename prefix: '<task>_<model>.*' (lowercased)."""
    left = prefix.split(".")[0]
    parts = left.split("_")
    task = parts[0].lower()
    model = parts[1].lower() if len(parts) > 1 else "unknown"
    return task, model

runs = []
for manifest_path in sorted(FINAL_ROOT.glob("*.manifest.json")):
    prefix = manifest_path.name.replace(".manifest.json", "")
    results_path = manifest_path.with_name(prefix + ".results.jsonl")
    boot_path = manifest_path.with_name(prefix + ".manifest.bootstrap_indices.npz")

    # These files are required for deterministic evaluation with stored bootstrap resamples.
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
# Categories for this ablation
# Core: scalar-only probe vs vector probe (GE)
# Optional appendix category: individual scalar signals
# ============================================================
CORE_CATS = ["egh_probe_scalar_only", "egh_probe_ge"]
CORE_PRETTY = {
    "egh_probe_scalar_only": "Scalar-only\nprobe",
    "egh_probe_ge": "Vector probe\n(G+E)",
}

# Optional: individual scalar metrics (if you want Appendix evidence)
SCALAR_METRICS = ["egh_grad_norm", "egh_emb_diff", "egh_kl", "egh_ce", "egh_entropy"]
SCALAR_PRETTY = {
    "egh_grad_norm": "GradNorm",
    "egh_emb_diff": "EmbDiff",
    "egh_kl": "KL",
    "egh_ce": "CE",
    "egh_entropy": "Entropy",
}


# ============================================================
# Compute metrics per run
# ============================================================
records = []
for task, model, manifest_path, results_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        continue

    y_key = find_label_key(rows[0])
    # Label convention: y is binary (0/1); downstream metrics treat 1 as the "positive" class.
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    score_dicts = [extract_scores(r) for r in rows]
    # Invariant: restrict to score keys present for every row to preserve alignment with y.
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}

    missing_core = [k for k in CORE_CATS if k not in S]
    if missing_core:
        print(f"[WARN] {results_path.name}: missing keys {missing_core}; skipping this run.")
        continue

    BOOT_KEY_BY_CAT = {
        "egh_probe_ge": "egh_ge",
        "egh_probe_scalar_only": "egh_scalar",
    }

    # Core ablation categories
    for cat in CORE_CATS:
        s_raw = S[cat]
        # Polarity is standardized per-run to avoid mixing "higher = better" vs "higher = worse" scores.
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        boot_key = BOOT_KEY_BY_CAT.get(cat, None)
        boot_idx = load_bootstrap_indices(boot_path, key=boot_key)

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "category": cat,
            "category_group": "core",
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

    # Optional scalar metrics (for appendix detail; only if present)
    for cat in SCALAR_METRICS:
        if cat not in S:
            continue
        s_raw = S[cat]
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        boot_idx_scalar = load_bootstrap_indices(boot_path, key="egh_scalar")

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx_scalar, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx_scalar, alpha=0.05)
        
        records.append({
            "task": task,
            "model": model,
            "category": cat,
            "category_group": "scalar_metrics",
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
out_csv = OUT_DIR / "analysis_egh_scalar_vs_vector_metrics.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)

if df.empty:
    raise RuntimeError("No rows computed. Check final results contain expected EGH probe keys.")


# ============================================================
# Plotting helpers (Fix: reindex, shaded CI band, black whiskers)
# ============================================================
# Canonical ordering improves visual comparability across repeated figure generations.
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])


def plot_overlay(df_sub: pd.DataFrame, cats: list, pretty_map: dict,
                 metric: str, y_lim, title: str, outpath: Path, xtick_rotation=0):
    """Overlay line plots per model with shaded 95% CIs, faceted by task."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    x = np.arange(len(cats), dtype=float)

    for ax, task in zip(axes, tasks):
        for model in models:
            sub = df_sub[(df_sub["task"] == task) & (df_sub["model"] == model)].copy()
            if sub.empty:
                continue

            # Alignment invariant: reindex to requested category order (missing cats become NaN).
            sub = sub.set_index("category").reindex(cats).reset_index()

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

            # Only shade CI where all three values are finite; NaNs propagate from missing/degenerate bootstraps.
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.15)

            ax.plot(x, y, marker="o", label=MODEL_PRETTY.get(model, model))
            # Whiskers are drawn in black to avoid ambiguous mapping to model color.
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=ERRORBAR_CAPSIZE, elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)

        ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_map.get(c, c) for c in cats], rotation=xtick_rotation, ha="center")
        ax.set_ylim(*y_lim)
        ax.set_title(TASK_PRETTY.get(task, task))
        ax.set_ylabel(ylabel)

    axes[0].legend(frameon=False, title="Model")
    fig.suptitle(title, y=0.995, fontsize=plt.rcParams["figure.titlesize"])
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.subplots_adjust(wspace=0.18)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)



def plot_bars_matrix(df_sub: pd.DataFrame, cats: list, pretty_map: dict,
                     metric: str, y_lim, outstem: str):
    """
    Plot a fixed 2x2 grid (task x model) with bars and 95% CI whiskers.

    Cells are ordered as:
      cols = [mistral, biomistral]
      rows = [medqa, pubmedqa]
    Missing combinations are left blank (no imputation).
    """

    tasks_order = [t for t in ["medqa", "pubmedqa"] if t in tasks]
    models_order = [m for m in ["mistral", "biomistral"] if m in models]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10.0, 9.2),
        sharey=True
    )
    axes = np.array(axes)
    x = np.arange(len(cats), dtype=float)

    for r, task in enumerate(tasks_order):
        for c, model in enumerate(models_order):

            ax = axes[r, c]
            sub = df_sub[(df_sub["task"] == task) & (df_sub["model"] == model)].copy()

            if sub.empty:
                ax.set_axis_off()
                continue

            # Alignment invariant: reindex to requested category order for consistent x positions.
            sub = sub.set_index("category").reindex(cats).reset_index()

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

            ax.bar(x, y)
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black", elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)

            ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)

            ax.set_xticks(x)
            ax.set_xticklabels([pretty_map.get(cat, cat) for cat in cats])
            ax.set_ylim(*y_lim)

            ax.set_title(
                f"{TASK_PRETTY.get(task, task)} — "
                f"{MODEL_PRETTY.get(model, model)}"
            )

            if c == 0:
                ax.set_ylabel(ylabel)

            # Value labels are placed above the upper CI whisker for readability at tight y-limits.
            add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

    fig.suptitle(
        f"Ablation: EGH Scalar vs Vector — "
        f"{'AUROC' if metric == 'auroc' else 'Spearman ρ'} ± 95% CI",
        y=0.97,
        fontsize=plt.rcParams["figure.titlesize"],
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.subplots_adjust(hspace=0.35, wspace=0.12)

    safe_savefig(fig, OUT_DIR / f"{outstem}_{metric}_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", OUT_DIR / f"{outstem}_{metric}_matrix.pdf")
    

# ============================================================
# Core plots: Scalar-only probe vs Vector probe (G+E)
# ============================================================
df_core = df[df["category_group"] == "core"].copy()

plot_overlay(
    df_sub=df_core,
    cats=CORE_CATS,
    pretty_map=CORE_PRETTY,
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="Ablation: EGH Scalar-only vs Vector Probe — AUROC ± 95% CI",
    outpath=OUT_DIR / "fig_ablation_egh_scalar_vs_vector_auroc_overlay.pdf",
    xtick_rotation=0,
)

plot_overlay(
    df_sub=df_core,
    cats=CORE_CATS,
    pretty_map=CORE_PRETTY,
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="Ablation: EGH Scalar-only vs Vector Probe — Spearman ρ ± 95% CI",
    outpath=OUT_DIR / "fig_ablation_egh_scalar_vs_vector_spearman_overlay.pdf",
    xtick_rotation=0,
)


plot_bars_matrix(
    df_sub=df_core,
    cats=CORE_CATS,
    pretty_map=CORE_PRETTY,
    metric="auroc",
    y_lim=AUROC_YLIM,
    outstem="fig_ablation_egh_scalar_vs_vector",
)

plot_bars_matrix(
    df_sub=df_core,
    cats=CORE_CATS,
    pretty_map=CORE_PRETTY,
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    outstem="fig_ablation_egh_scalar_vs_vector",
)


# ============================================================
# Optional appendix plots: individual scalar metrics (if desired)
# ============================================================
df_scalar = df[df["category_group"] == "scalar_metrics"].copy()
if not df_scalar.empty:
    scalar_present = [k for k in SCALAR_METRICS if k in set(df_scalar["category"])]

    plot_overlay(
        df_sub=df_scalar,
        cats=scalar_present,
        pretty_map=SCALAR_PRETTY,
        metric="auroc",
        y_lim=AUROC_YLIM,
        title="Appendix: EGH Scalar Signals — AUROC ± 95% CI",
        outpath=OUT_DIR / "fig_appendix_egh_scalar_signals_auroc_overlay.pdf",
        xtick_rotation=20,
    )

    plot_overlay(
        df_sub=df_scalar,
        cats=scalar_present,
        pretty_map=SCALAR_PRETTY,
        metric="spearman",
        y_lim=SPEARMAN_YLIM,
        title="Appendix: EGH Scalar Signals — Spearman ρ ± 95% CI",
        outpath=OUT_DIR / "fig_appendix_egh_scalar_signals_spearman_overlay.pdf",
        xtick_rotation=20,
    )

print("[OK] EGH scalar vs vector ablation done. Outputs in:", OUT_DIR)