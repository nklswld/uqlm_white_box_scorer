# phase_2_medical/analysis/ablations/analyze_hidden_pooling.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: EXACTLY consistent with phase2_figures.py  (Fix 1)
# ============================================================
FONT_SCALE = 1.35  # keep consistent

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
    "figure.titlesize": int(14.5 * FONT_SCALE),

    "axes.titlepad": 12,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

AUROC_YLIM = (0.45, 0.80)
SPEARMAN_YLIM = (-0.05, 0.60)

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}
def pretty_score(k: str) -> str:
    kk = str(k).lower()
    return SCORE_PRETTY.get(kk, kk)


# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
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
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

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
    # canonical
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    # fallback: pick numeric fields that look like scores
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)
    return scores

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

    # handle pathological cases where ALL bootstrap samples were invalid
    if aucs.size == 0:
        return np.nan, np.nan, np.nan

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

    # handle pathological cases where ALL bootstrap samples were invalid
    if rhos.size == 0:
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
OUT_DIR = ROOT / "outputs" / "figs" / "ablations" / "hidden_pooling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # derive task/model/pooling from folders
    # .../hidden_pooling/medqa_biomistral/last_answer/<file>.manifest.json
    try:
        pooling = manifest_path.parent.name
        task_model = manifest_path.parent.parent.name
        task = task_model.split("_")[0].lower()
        model = task_model.split("_")[1].lower()
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

    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    score_dicts = [extract_scores(r) for r in rows]
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted([str(k).lower() for k in keys])

    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
    boot_idx = load_bootstrap_indices(boot_path)

    for score_key, s_raw in S.items():
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "pooling": pooling,
            "score_key": score_key,
            "direction": float(direction),
            "N": int(len(y)),
            "pos_rate": float(y.mean()),

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
    # fallback: if your key is different, show available keys
    print("[WARN] No rows with score_key='hidden_probe_oof'. Available score_key values:")
    print(sorted(df["score_key"].unique().tolist()))
    # still proceed with all keys, but this ablation is usually Hidden-only
    df_hid = df.copy()

# stable pooling order: prefer known order, then alphabetical remainder
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
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        x = np.arange(len(poolings), dtype=float)

        for model in models:
            sub = df_hid[(df_hid["task"] == task) & (df_hid["model"] == model)].copy()
            if sub.empty:
                continue

            # Fix 4: reindex to full pooling list to avoid misalignment
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

            # Fix 5: CI band (with NaN-safe masking)
            mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if mask.any():
                ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.15)

            # line + markers
            ax.plot(x, y, marker="o", label=MODEL_PRETTY.get(model, model))

            # black whiskers (phase2-like)
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, ecolor="black")

            # annotate best pooling per model (keep your original idea; NaN-safe)
            if np.isfinite(y).any():
                best_i = int(np.nanargmax(y))
                ax.text(
                    x[best_i],
                    y[best_i] + 0.02 * (y_lim[1] - y_lim[0]),
                    f"{y[best_i]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_LABEL_FONTSIZE,
                )

        ax.axhline(hline, linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(poolings)))
        ax.set_xticklabels(poolings, rotation=20, ha="right")
        ax.set_ylim(*y_lim)
        ax.set_title(TASK_PRETTY.get(task, task))
        ax.set_ylabel(ylabel)

    axes[0].legend(frameon=False, title="Model")
    fig.suptitle(title, y=1.02, fontsize=plt.rcParams["figure.titlesize"])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


# ============================================================
# Bar plots: per task × model (Fix 4 applied here too)
# - bars = poolings
# - black whiskers + value labels (already phase2-like)
# ============================================================
def plot_bars(metric: str, y_lim, outstem: str):
    for task in tasks:
        for model in models:
            sub = df_hid[(df_hid["task"] == task) & (df_hid["model"] == model)].copy()
            if sub.empty:
                continue

            # Fix 4: reindex to full pooling list to keep bars aligned and stable
            sub = sub.set_index("pooling").reindex(poolings).reset_index()

            x = np.arange(len(poolings), dtype=float)

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

            fig, ax = plt.subplots(figsize=(10.8, 4.9))
            ax.bar(x, y)
            ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", capsize=3, ecolor="black")
            ax.axhline(hline, linestyle="--", linewidth=1)

            ax.set_xticks(x)
            ax.set_xticklabels(poolings, rotation=20, ha="right")
            ax.set_ylim(*y_lim)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{TASK_PRETTY.get(task, task)} — {MODEL_PRETTY.get(model, model)}")

            add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

            safe_savefig(fig, OUT_DIR / f"{outstem}_{metric}_{task}_{model}.pdf", bbox_inches="tight")
            plt.close(fig)
            print("Wrote:", OUT_DIR / f"{outstem}_{metric}_{task}_{model}.pdf")


# Overlay figures (story-like; per task; lines = models; with CI band)
plot_overlay(
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="Ablation: Hidden Probe Pooling — AUROC ± 95% CI (Hidden; by task; lines = models)",
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_auroc_overlay.pdf",
)
plot_overlay(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="Ablation: Hidden Probe Pooling — Spearman ρ ± 95% CI (Hidden; by task; lines = models)",
    outpath=OUT_DIR / "fig_ablation_hidden_pooling_spearman_overlay.pdf",
)

# Detailed bars (per task × model)
plot_bars(metric="auroc", y_lim=AUROC_YLIM, outstem="fig_ablation_hidden_pooling")
plot_bars(metric="spearman", y_lim=SPEARMAN_YLIM, outstem="fig_ablation_hidden_pooling")

print("[OK] Hidden pooling ablation done. Outputs in:", OUT_DIR)