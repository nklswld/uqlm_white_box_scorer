"""
Analyze hidden-layer probe ablations and generate publication-ready summary figures.

Inputs: per-run *.manifest.json (task/model metadata), matching *.results.jsonl (per-example labels/scores),
and matching *.manifest.bootstrap_indices.npz (bootstrap resampling indices and (optionally) kept-example masks).
Outputs: a CSV of per-(task, model, layer) metrics with 95% bootstrap CIs, plus PDF figures (overlays + 2x2 matrices).
Reproducibility: all estimates are deterministic given saved bootstrap indices; no RNG is used in this script.
"""

# phase_2_medical/analysis/ablations/analyze_hidden_layers.py
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import matplotlib as mpl


# ============================================================
# Plot styling (kept consistent with phase2_figures.py for cross-figure comparability)
# ============================================================
FONT_SCALE = 1.5 # reviewer-facing typography scaling; keep in sync with related figure scripts

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

    # Slightly thicker axes/ticks for print/PDF legibility.
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    
    # Embed editable text in vector backends (important for camera-ready figure tweaks).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)


# ============================================================
# Paths (script-local, repository-relative)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
ABL_DIR = ROOT / "outputs" / "ablations" / "hidden_layers"
FIGS_DIR = ROOT / "outputs" / "figures_tables" / "ablations" / "hidden_layers"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Single consolidated table for downstream aggregation / plotting (one row per task×model×layer).
OUT_CSV = FIGS_DIR / "analysis_hidden_layers_metrics.csv"
ABL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot constants (fixed y-limits enable visual comparison across panels/tasks/models)
# ============================================================
AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.10, 0.70)

# ---------------------------------------------------------------------
# Plot styling knobs (global, for print/readability)
# ---------------------------------------------------------------------
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Ternary)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

# Hidden-layer sweep convention: score key used across runs for the probe's per-example output.
PRIMARY_SCORE_KEY = "hidden_probe_oof"


# ============================================================
# Robust save helper (Windows PDF file locks)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save figure to PDF; if outpath is locked, fall back to a versioned *_vK.pdf filename."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        # NOTE: potential issue: writing *_vK.pdf changes the artifact name expected by downstream scripts.
        stem = outpath.stem
        suffix = outpath.suffix
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
# Helpers
# ============================================================
def load_jsonl(path: Path):
    """Load JSONL file into a list of dicts; ignores blank lines to tolerate trailing newlines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def np_load_array_by_preference(npz_path: Path, preferred_keys=None):
    """
    Load a single array from NPZ using a preferred key order, then common conventions, else first entry.

    NOTE: potential issue: "first entry" fallback depends on NPZ key ordering and may vary across writers.
    """
    z = np.load(npz_path, allow_pickle=True)
    preferred_keys = list(preferred_keys or [])
    for k in preferred_keys:
        if k in z.files:
            return z[k]
    for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if k in z.files:
            return z[k]
    return z[z.files[0]]

def load_bootstrap_indices(boot_path: Path, preferred_keys=None):
    """Load bootstrap resample indices as int array with shape [B, N] (B resamples over N examples)."""
    arr = np_load_array_by_preference(boot_path, preferred_keys=preferred_keys)
    # Some pipelines save an object array of per-resample index vectors; stack to [B, N].
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)

def find_label_key(example: dict):
    """Heuristically locate the binary label field in a result row (first match wins)."""
    # Convention across pipelines: label may be stored under any of these common keys.
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")


def extract_scores(example: dict):
    """Extract a score dict from a row; fall back to scalar numeric 'hidden*' fields if needed."""
    # Convention: nested score dicts reduce key collisions with other per-example metadata.
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    # Fallback: collect scalar numeric fields with names indicating hidden-layer probe outputs.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            if "hidden" in kk:
                scores[kk] = float(v)
    return scores


def infer_task_model_from_manifest(manifest_path: Path):
    """Infer (task, model) from manifest.json; model uses a 'bio' substring convention."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(m.get("task", "")).lower()
    config = m.get("config", {})
    model_name = str(config.get("model_name", "")).lower()
    # Heuristic: "bio" in the model name selects biomistral vs mistral.
    model = "biomistral" if "bio" in model_name else "mistral"
    return task, model


def infer_layer(manifest_path: Path):
    """
    Infer hidden layer index via (1) path convention, else (2) manifest config, else filename fallback.

    NOTE: potential issue: conflicting signals (e.g., folder vs config) are not checked for consistency.
    """
    # (1) Path-based inference (preferred: explicit layer_* folder names are unambiguous).
    for part in manifest_path.parts[::-1]:
        m = re.match(r"layer[_\-]?(\d+)", str(part).lower())
        if m:
            return int(m.group(1))

    # (2) Manifest-based inference (supports alternative config schemas across experiments).
    mjson = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = mjson.get("config", {})
    for k in ["hidden_layer", "hidden_layer_idx", "layer", "layer_idx"]:
        if k in cfg:
            try:
                return int(cfg[k])
            except Exception:
                # NOTE: potential issue: non-integer layer values are ignored, potentially masking misconfigured runs.
                pass

    # Fallback: try parse digits anywhere in filename.
    m = re.search(r"layer[_\-]?(\d+)", manifest_path.name.lower())
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not infer layer for: {manifest_path}")


def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and flip score polarity if needed so AUROC >= 0.5; returns (auroc, direction)."""
    au = roc_auc_score(y, s)
    # Polarity convention: direction=-1 means scores were sign-flipped for consistent "higher is better".
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """
    Bootstrap AUROC mean and central (1-alpha) CI using precomputed resample indices.

    NOTE: skipping single-class resamples may affect CI width under strong class imbalance and should be considered when interpreting results.
    """
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: AUROC undefined if only one class present.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))
    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        # All resamples degenerate -> CI undefined; propagate NaNs into summary tables/figures.
        return np.nan, np.nan, np.nan
    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi


def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """
    Bootstrap Spearman rho mean and central (1-alpha) CI using precomputed resample indices.

    NOTE: potential issue: Spearman on binary y reduces to rank-biserial-like behavior; interpret accordingly.
    """
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: correlation undefined if only one class present.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        # Some resamples can yield NaN (e.g., ties/constant ranks); skip to avoid contaminating quantiles.
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


def add_value_labels_above_ci(ax, x_positions, y_values, yerr_high,
                             fmt="{:.3f}", fontsize=None, pad_frac=0.02):
    """Annotate plotted estimates with numeric labels placed above the upper CI whisker."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span

    for x, y, eh in zip(x_positions, y_values, yerr_high):
        # Skip missing points (propagates through NaN-based masking upstream).
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        top = y + (0.0 if eh is None else float(eh))
        ax.text(float(x), float(top) + pad, fmt.format(float(y)),
                ha="center", va="bottom", fontsize=fontsize)


# ============================================================
# Collect Hidden-Layer ablation runs
# Expected per run (same naming logic as phase2):
#   *.manifest.json
#   sameprefix.results.jsonl
#   sameprefix.manifest.bootstrap_indices.npz
# ============================================================
print("ROOT =", ROOT)
print("ABL_DIR =", ABL_DIR)
print("ABL_DIR exists =", ABL_DIR.exists())

if not ABL_DIR.exists():
    raise FileNotFoundError(f"Hidden-layer ablation directory not found: {ABL_DIR}")

runs = []
for manifest_path in sorted(ABL_DIR.rglob("*.manifest.json")):
    results_path = manifest_path.with_suffix("").with_suffix(".results.jsonl")
    boot_path = manifest_path.with_suffix("").with_suffix(".manifest.bootstrap_indices.npz")

    # Skip incomplete runs (prevents silent layer-coverage bias in summary tables/figures).
    if not results_path.exists():
        print("[WARN] Missing results for", manifest_path, "expected:", results_path.name)
        continue
    if not boot_path.exists():
        print("[WARN] Missing bootstrap npz for", manifest_path, "expected:", boot_path.name)
        continue

    task, model = infer_task_model_from_manifest(manifest_path)
    layer = infer_layer(manifest_path)
    runs.append((task, model, layer, results_path, manifest_path, boot_path))

print("Found runs:", len(runs))
if len(runs) == 0:
    raise RuntimeError("No hidden-layer runs found. Check folder structure / file naming.")


# ============================================================
# Compute metrics per (task, model, layer)
# ============================================================
records = []
for task, model, layer, results_path, manifest_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        # Empty results contribute no statistics; keep quiet to avoid noisy logs across large sweeps.
        continue

    # Label-key is inferred from the first row; assumes schema is consistent within a run.
    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # --- Hidden-probe values: read from top-level field (NaN-safe) ---
    # Works even if some rows contain None (-> NaN) due to strict=False dropping examples upstream.
    score_key = PRIMARY_SCORE_KEY

    s_full = np.array(
        [
            (np.nan if (r.get(score_key, None) is None) else float(r.get(score_key)))
            for r in rows
        ],
        dtype=float,
    )

    # --- Use kept subset if available in NPZ (preferred, matches how hidden bootstraps were generated) ---
    z = np.load(boot_path, allow_pickle=True)
    if "hidden_kept_indices" in z.files and "hidden" in z.files:
        # Invariant: kept indices define the exact evaluation set used when precomputing bootstrap indices.
        kept = z["hidden_kept_indices"].astype(int)
        s_raw = s_full[kept]
        y_use = y[kept]
        boot_idx = load_bootstrap_indices(boot_path, preferred_keys=["hidden"])
    else:
        # Fallback: infer kept via finite mask (less ideal, but still deterministic).
        # NOTE: potential issue: mask-derived "kept" may not match the set used to generate boot_idx in older runs.
        mask = np.isfinite(s_full)
        s_raw = s_full[mask]
        y_use = y[mask]
        boot_idx = load_bootstrap_indices(boot_path)  # last resort fallback

    # Guard: if everything got dropped, skip this run.
    if y_use.size == 0 or np.unique(y_use).size < 2:
        print(f"[WARN] Hidden score empty/degenerate after masking for {results_path.name}")
        continue

    # Polarity convention is estimated on the actually-used subset, then applied consistently for all metrics.
    au, direction = auroc_with_best_direction(y_use, s_raw)
    s = s_raw * direction

    # CIs are computed from saved bootstrap indices (deterministic; no RNG).
    au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y_use, s, boot_idx, alpha=0.05)
    rho_mean, rho_lo, rho_hi = bootstrap_spearman_ci_from_indices(y_use, s, boot_idx, alpha=0.05)

    records.append({
        "task": task,
        "model": model,
        "layer": int(layer),
        "score_key": score_key,
        "direction": float(direction),

        "auroc": float(au),
        "auroc_boot_mean": float(au_mean),
        "auroc_ci95_lo": float(au_lo),
        "auroc_ci95_hi": float(au_hi),

        "spearman_rho_boot_mean": float(rho_mean),
        "spearman_ci95_lo": float(rho_lo),
        "spearman_ci95_hi": float(rho_hi),

        "N": int(len(y_use)),
        "pos_rate": float(y_use.mean()),

        "results_file": str(results_path),
        "manifest_file": str(manifest_path),
        "boot_file": str(boot_path),
    })

df = pd.DataFrame(records)
df = df.sort_values(["task", "model", "layer"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)


# ============================================================
# Plot helpers: line + CI
# ============================================================
def _plot_line_ci(ax, x, y, lo, hi, label):
    """Plot point estimates with shaded CI band and high-contrast error bars."""
    ax.plot(x, y, marker="o", label=label)
    ax.fill_between(x, lo, hi, alpha=0.15)
    # Error bars are drawn in black to remain legible across styles/exports.
    yerr_low = y - lo
    yerr_high = hi - y
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black", 
                elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)
    

def plot_metric_matrix(df_all: pd.DataFrame, metric: str, outpath: Path):
    """
    Render a 2x2 grid over tasks (rows) and models (cols) for a given metric, with per-layer 95% CIs.

    NOTE: fixed task/model order and fixed y-limits are intentional for cross-panel comparability.
    """
    tasks_order = ["medqa", "pubmedqa"]
    models_order = ["mistral", "biomistral"]

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 9.2), sharey=True)
    axes = np.array(axes)

    for r, task in enumerate(tasks_order):
        for c, model in enumerate(models_order):
            ax = axes[r, c]
            sub = df_all[(df_all["task"] == task) & (df_all["model"] == model)].copy()
            if sub.empty:
                ax.set_axis_off()
                continue

            # Invariant: x is layer index; per-panel sort ensures monotonic x even if input CSV is shuffled.
            sub = sub.sort_values("layer")
            x = sub["layer"].to_numpy(dtype=int)

            if metric == "auroc":
                y = sub["auroc_boot_mean"].to_numpy(dtype=float)
                lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # chance-level reference for AUROC
                ax.set_ylim(*AUROC_YLIM)
                ylabel = "AUROC"
            else:
                y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # null association reference
                ax.set_ylim(*SPEARMAN_YLIM)
                ylabel = "Spearman ρ (bootstrap mean)"

            _plot_line_ci(ax, x, y, lo, hi, label=None)
            

            # Enforce global y-limits after adding artists (prevents autoscale drift across panels).
            if metric == "auroc":
                ax.set_ylim(*AUROC_YLIM)
            else:
                ax.set_ylim(*SPEARMAN_YLIM)

            ax.set_xlabel("Hidden layer index")
            if c == 0:
                ax.set_ylabel(ylabel)

            task_title = TASK_PRETTY.get(task, task)
            task_title = task_title.replace("(Ternary)", "(Ternary)\n")

            ax.set_title(
                f"{task_title} — {MODEL_PRETTY.get(model, model)}"
            )

    fig.suptitle(
        "Hidden Layer Sweep — " + ("AUROC ± 95% CI" if metric == "auroc" else "Spearman ρ ± 95% CI"),
        y=0.975,
        fontsize=mpl.rcParams["figure.titlesize"] * 1.13,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.subplots_adjust(hspace=0.60, wspace=0.12)

    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath



def plot_overlay_auroc_two_tasks(df_all: pd.DataFrame):
    """AUROC-Overlay für beide Tasks nebeneinander: MedQA links, PubMedQA rechts."""
    tasks_order = ["medqa", "pubmedqa"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    axes = np.array(axes)

    for ax, task in zip(axes, tasks_order):
        df_task = df_all[df_all["task"] == task].copy()

        for model in ["mistral", "biomistral"]:
            sub = df_task[df_task["model"] == model].sort_values("layer")
            if len(sub) == 0:
                continue

            x = sub["layer"].to_numpy(dtype=int)
            y = sub["auroc_boot_mean"].to_numpy(dtype=float)
            lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
            hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)

            _plot_line_ci(ax, x, y, lo, hi, label=MODEL_PRETTY.get(model, model))

        ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
        ax.set_ylim(*AUROC_YLIM)
        ax.set_xlabel("Hidden layer index")
        ax.set_title(TASK_PRETTY.get(task, task))

    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, title="Model")

    fig.suptitle(
        "Hidden Layer Sweep — AUROC ± 95% CI",
        y=0.98,
        fontsize=mpl.rcParams["figure.titlesize"] * 1.13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.18)

    out = FIGS_DIR / "fig_ablation_hidden_layers_auroc_overlay.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overlay_spearman_two_tasks(df_all: pd.DataFrame):
    """Spearman-Overlay für beide Tasks nebeneinander: MedQA links, PubMedQA rechts."""
    tasks_order = ["medqa", "pubmedqa"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    axes = np.array(axes)

    for ax, task in zip(axes, tasks_order):
        df_task = df_all[df_all["task"] == task].copy()

        for model in ["mistral", "biomistral"]:
            sub = df_task[df_task["model"] == model].sort_values("layer")
            if len(sub) == 0:
                continue

            x = sub["layer"].to_numpy(dtype=int)
            y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
            lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
            hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)

            _plot_line_ci(ax, x, y, lo, hi, label=MODEL_PRETTY.get(model, model))

        ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
        ax.set_ylim(*SPEARMAN_YLIM)
        ax.set_xlabel("Hidden layer index")
        ax.set_title(TASK_PRETTY.get(task, task))

    axes[0].set_ylabel("Spearman ρ (bootstrap mean)")
    axes[0].legend(frameon=False, title="Model")

    fig.suptitle(
        "Hidden Layer Sweep — Spearman ρ ± 95% CI",
        y=0.98,
        fontsize=mpl.rcParams["figure.titlesize"] * 1.13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.18)

    out = FIGS_DIR / "fig_ablation_hidden_layers_spearman_overlay.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Generate figures
# ============================================================
written = []

# Both tasks displayed side by side within a single overlay figure
written.append(plot_overlay_auroc_two_tasks(df))
written.append(plot_overlay_spearman_two_tasks(df))

# 2x2 matrices emphasize task/model structure at fixed metric scale.
written.append(
    plot_metric_matrix(
        df_all=df,
        metric="auroc",
        outpath=FIGS_DIR / "fig_ablation_hidden_layers_auroc_matrix.pdf",
    )
)

written.append(
    plot_metric_matrix(
        df_all=df,
        metric="spearman",
        outpath=FIGS_DIR / "fig_ablation_hidden_layers_spearman_matrix.pdf",
    )
)

print("[OK] Hidden layer sweep figures written to:", FIGS_DIR)
for p in written:
    print("  -", p)