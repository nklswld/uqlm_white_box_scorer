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
# Style: keep consistent with phase2_figures.py
# ============================================================
FONT_SCALE = 1.35  # keep same default; adjust only if needed

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

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)


# ============================================================
# Paths
# This script lives under: phase_2_medical/analysis/ablations/
# ROOT should be:          phase_2_medical/
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
ABL_DIR = ROOT / "outputs" / "ablations" / "hidden_layers"
FIGS_DIR = ROOT / "outputs" / "figs" / "ablations" / "hidden_layers"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Where to store the summary CSV for this ablation
OUT_CSV = FIGS_DIR / "analysis_hidden_layers_metrics.csv"
ABL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot constants
# ============================================================
AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.10, 0.70)

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

# We analyze the Hidden probe layer sweep.
# In the layer folders, the score is usually still "hidden_probe_oof".
PRIMARY_SCORE_KEY = "hidden_probe_oof"


# ============================================================
# Robust save helper (Windows PDF file locks)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """
    Save a figure as PDF robustly on Windows where an open PDF can lock the file.
    If the target is locked, write to *_v2.pdf, *_v3.pdf, ...
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
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
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def np_load_first_array(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if k in z.files:
            return z[k]
    return z[z.files[0]]


def load_bootstrap_indices(boot_path: Path):
    arr = np_load_first_array(boot_path)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)


def find_label_key(example: dict):
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")


def extract_scores(example: dict):
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            if "hidden" in kk:
                scores[kk] = float(v)
    return scores


def infer_task_model_from_manifest(manifest_path: Path):
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(m.get("task", "")).lower()
    config = m.get("config", {})
    model_name = str(config.get("model_name", "")).lower()
    model = "biomistral" if "bio" in model_name else "mistral"
    return task, model


def infer_layer(manifest_path: Path):
    """
    Robustly infer the layer index:
    1) from path components like .../layer_16/...
    2) from manifest config keys (hidden_layer, layer, hidden_layer_idx, etc.)
    """
    # (1) path-based inference
    for part in manifest_path.parts[::-1]:
        m = re.match(r"layer[_\-]?(\d+)", str(part).lower())
        if m:
            return int(m.group(1))

    # (2) manifest-based inference
    mjson = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = mjson.get("config", {})
    for k in ["hidden_layer", "hidden_layer_idx", "layer", "layer_idx"]:
        if k in cfg:
            try:
                return int(cfg[k])
            except Exception:
                pass

    # fallback: try parse digits anywhere in filename
    m = re.search(r"layer[_\-]?(\d+)", manifest_path.name.lower())
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not infer layer for: {manifest_path}")


def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))
    aucs = np.asarray(aucs, dtype=float)
    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi


def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        if pd.isna(rho):
            continue
        rhos.append(float(rho))
    rhos = np.asarray(rhos, dtype=float)
    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi


def add_value_labels_above_ci(ax, x_positions, y_values, yerr_high,
                             fmt="{:.3f}", fontsize=None, pad_frac=0.02):
    """
    Place labels above the CI whisker (y + yerr_high) for each point/bar.
    """
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span

    for x, y, eh in zip(x_positions, y_values, yerr_high):
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
        continue

    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    score_dicts = [extract_scores(r) for r in rows]
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted(keys)

    # Try primary key first, otherwise fallback to any hidden-like key
    score_key = None
    lower_keys = [k.lower() for k in keys]
    if PRIMARY_SCORE_KEY in lower_keys:
        score_key = PRIMARY_SCORE_KEY
    else:
        candidates = [k for k in lower_keys if "hidden" in k]
        if len(candidates) == 0:
            print(f"[WARN] No hidden-like score found in {results_path.name}. Keys={keys}")
            continue
        score_key = candidates[0]

    s_raw = np.array([d[score_key] for d in score_dicts], dtype=float)
    au, direction = auroc_with_best_direction(y, s_raw)
    s = s_raw * direction

    boot_idx = load_bootstrap_indices(boot_path)
    au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05)

    rho_mean, rho_lo, rho_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05)

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

        "N": int(len(y)),
        "pos_rate": float(y.mean()),

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
    ax.plot(x, y, marker="o", label=label)
    ax.fill_between(x, lo, hi, alpha=0.15)
    # errorbars in black for contrast
    yerr_low = y - lo
    yerr_high = hi - y
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=3, ecolor="black")



def plot_metric_matrix(df_all: pd.DataFrame, metric: str, outpath: Path):
    """
    2x2 matrix:
      cols = [mistral, biomistral]
      rows = [medqa, pubmedqa]
    metric: "auroc" or "spearman"
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

            sub = sub.sort_values("layer")
            x = sub["layer"].to_numpy(dtype=int)

            if metric == "auroc":
                y = sub["auroc"].to_numpy(dtype=float)
                lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                ax.axhline(0.5, linestyle="--", linewidth=1)
                ax.set_ylim(*AUROC_YLIM)
                ylabel = "AUROC"
            else:
                y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                ax.axhline(0.0, linestyle="--", linewidth=1)
                ax.set_ylim(*SPEARMAN_YLIM)
                ylabel = "Spearman ρ (bootstrap mean)"

            _plot_line_ci(ax, x, y, lo, hi, label=None)
            

            # enforce again after all artists
            if metric == "auroc":
                ax.set_ylim(*AUROC_YLIM)
            else:
                ax.set_ylim(*SPEARMAN_YLIM)

            ax.set_xlabel("Hidden layer index")
            if c == 0:
                ax.set_ylabel(ylabel)

            ax.set_title(f"{TASK_PRETTY.get(task, task)} — {MODEL_PRETTY.get(model, model)}")

    fig.suptitle(
        "Hidden Layer Sweep — " + ("AUROC ± 95% CI" if metric == "auroc" else "Spearman ρ ± 95% CI"),
        y=0.97,
        fontsize=mpl.rcParams["figure.titlesize"],
    )

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.subplots_adjust(hspace=0.45, wspace=0.12)

    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath



def plot_task_overlay_auroc(df_task: pd.DataFrame, task: str):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    for model in ["mistral", "biomistral"]:
        sub = df_task[df_task["model"] == model].sort_values("layer")
        if len(sub) == 0:
            continue
        x = sub["layer"].to_numpy(dtype=int)
        y = sub["auroc"].to_numpy(dtype=float)
        lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
        hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
        _plot_line_ci(ax, x, y, lo, hi, label=MODEL_PRETTY.get(model, model))

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(*AUROC_YLIM)
    ax.set_xlabel("Hidden layer index")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Hidden Layer Sweep — {TASK_PRETTY.get(task, task)} (Model overlay)")

    ax.legend(frameon=False, title="Model")
    fig.subplots_adjust(top=0.88)

    out = FIGS_DIR / f"fig_ablation_hidden_layers_auroc_overlay_{task}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_task_overlay_spearman(df_task: pd.DataFrame, task: str):
    fig, ax = plt.subplots(figsize=(11.5, 5.2))

    for model in ["mistral", "biomistral"]:
        sub = df_task[df_task["model"] == model].sort_values("layer")
        if len(sub) == 0:
            continue
        x = sub["layer"].to_numpy(dtype=int)
        y = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
        lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
        hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
        _plot_line_ci(ax, x, y, lo, hi, label=MODEL_PRETTY.get(model, model))

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylim(*SPEARMAN_YLIM)
    ax.set_xlabel("Hidden layer index")
    ax.set_ylabel("Spearman ρ (bootstrap mean)")
    ax.set_title(f"Hidden Layer Sweep — {TASK_PRETTY.get(task, task)} (Spearman overlay)")

    ax.legend(frameon=False, title="Model")
    fig.subplots_adjust(top=0.88)

    out = FIGS_DIR / f"fig_ablation_hidden_layers_spearman_overlay_{task}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# Generate figures
# ============================================================
written = []

for task in sorted(df["task"].unique()):
    df_task = df[df["task"] == task].copy()

    # Overlay plot per task (AUROC)
    written.append(plot_task_overlay_auroc(df_task, task))

    # Overlay plot per task (Spearman)
    written.append(plot_task_overlay_spearman(df_task, task))

# 2x2 matrices (AUROC + Spearman)
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