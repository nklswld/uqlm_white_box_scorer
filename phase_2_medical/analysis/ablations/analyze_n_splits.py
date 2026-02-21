# phase_2_medical/analysis/ablations/analyze_n_splits.py
#
# Ablation: OOF Robustness over different n_splits (e.g., 3, 5, 10)
# Goal: quantify bias/variance trade-off; probe should not be overly sensitive to the fold count.
#
# Expected artifacts per setting (analog to other ablations):
#   *.manifest.json
#   corresponding *.results.jsonl
#   corresponding *.manifest.bootstrap_indices.npz   (or any *bootstrap_indices*.npz)
#
# Folder convention (recommended):
#   phase_2_medical/outputs/ablations/n_splits/<task>_<model>/n_splits_3/...
#   phase_2_medical/outputs/ablations/n_splits/<task>_<model>/n_splits_5/...
#   phase_2_medical/outputs/ablations/n_splits/<task>_<model>/n_splits_10/...
#
# Outputs:
#  - phase_2_medical/outputs/ablations/n_splits/analysis_n_splits_metrics.csv
#      (per-setting metrics with bootstrap CI, for LNTP/MTP/EGH/Hidden)
#  - phase_2_medical/outputs/ablations/n_splits/analysis_n_splits_summary.csv
#      (per task×model×score: mean/std across n_splits settings)
#  - phase_2_medical/outputs/figs/ablations/n_splits/
#      fig_ablation_n_splits_auroc_overlay.pdf
#      fig_ablation_n_splits_spearman_overlay.pdf
#      fig_ablation_n_splits_auroc_by_task_model.pdf (optional detailed panel)
#
# IMPORTANT: Styling and structure match phase2_figures.py and your other ablation scripts.

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: consistent with phase2_figures.py
# ============================================================
FONT_SCALE = 1.5  # keep consistent

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

AUROC_YLIM = (0.45, 0.85)
SPEARMAN_YLIM = (-0.10, 0.70)

# ---------------------------------------------------------------------
# Plot styling knobs (global, for print/readability)
# ---------------------------------------------------------------------
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}

MAIN_SCORES = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]


def pretty_score(k: str) -> str:
    kk = str(k).lower()
    return SCORE_PRETTY.get(kk, kk)


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # ./phase_2_medical

ABL_DIR = ROOT / "outputs" / "ablations" / "n_splits"
FIGS_DIR = ROOT / "outputs" / "figs" / "ablations" / "n_splits"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
ABL_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FIGS_DIR / "analysis_n_splits_metrics.csv"
OUT_CSV_SUMMARY = FIGS_DIR / "analysis_n_splits_summary.csv"


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


def load_bootstrap_indices(boot_path: Path, key: str | None = None):
    z = np.load(boot_path, allow_pickle=True)

    # If a specific key is requested and exists, use it.
    if key is not None and key in z.files:
        arr = z[key]
    else:
        # Backward-compatible fallback to common single-array conventions
        for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
            if k in z.files:
                arr = z[k]
                break
        else:
            # Last resort: first entry (legacy)
            arr = z[z.files[0]]

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)


def find_label_key(example: dict):
    for k in ["is_error", "label", "y", "target", "hallucinated", "is_hallucinated"]:
        if k in example:
            return k
    raise KeyError(f"Could not find label key in example keys: {sorted(example.keys())[:50]}")


def extract_scores(example: dict):
    skip = {"qid", "task", "label", "gold", "pred", "model_answer", "meta"}
    out = {}
    for k, v in example.items():
        if k in skip:
            continue
        if isinstance(v, (int, float, np.number)) and np.isfinite(v):
            out[str(k).lower()] = float(v)
    return out


def auroc_with_best_direction(y, s_raw):
    au = roc_auc_score(y, s_raw)
    if au < 0.5:
        return roc_auc_score(y, -s_raw), -1.0
    return au, +1.0


def bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05):
    vals = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss))
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return mean, lo, hi


def bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05):
    vals = []
    y = np.asarray(y)
    s = np.asarray(s)
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        if pd.isna(rho):
            continue
        vals.append(float(rho))
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return mean, lo, hi


def infer_task_model_from_manifest(manifest: dict, fallback_path: Path):
    task = manifest.get("task", None)
    if task is None:
        name = fallback_path.name.lower()
        if "medqa" in name:
            task = "medqa"
        elif "pubmedqa" in name:
            task = "pubmedqa"
    task = (task or "unknown").lower()

    cfg = manifest.get("config", {}) if isinstance(manifest.get("config", {}), dict) else {}
    model_name = manifest.get("model_name") or cfg.get("model_name") or manifest.get("model") or ""
    model = None
    mn = str(model_name).lower()
    if "biomistral" in mn:
        model = "biomistral"
    elif "mistral" in mn:
        model = "mistral"

    if model is None:
        name = fallback_path.name.lower()
        if "biomistral" in name:
            model = "biomistral"
        elif "mistral" in name:
            model = "mistral"

    model = model or "unknown"
    return task, model


def infer_n_splits(manifest: dict, manifest_path: Path):
    # Prefer manifest/config key if present
    cfg = manifest.get("config", {}) if isinstance(manifest.get("config", {}), dict) else {}
    for key in ["n_splits", "nsplits", "cv_splits"]:
        for src in (manifest, cfg):
            if key in src:
                try:
                    return int(src[key])
                except Exception:
                    pass

    # Common folder pattern: n_splits_5, nsplits_5, splits_5
    txt = str(manifest_path).lower()
    for pat in [r"n[_\-]?splits[_\-]?(\d+)", r"nsplits[_\-]?(\d+)", r"splits[_\-]?(\d+)"]:
        m = re.search(pat, txt)
        if m:
            return int(m.group(1))

    return -1


def find_runs(abl_root: Path):
    manifests = sorted(abl_root.rglob("*.manifest.json"))
    runs = []
    for mp in manifests:
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue

        task, model = infer_task_model_from_manifest(manifest, mp)
        n_splits = infer_n_splits(manifest, mp)

        stem = mp.name.replace(".manifest.json", "")
        # results
        candidates_results = [
            mp.with_name(f"{stem}.results.jsonl"),
            mp.with_name(f"{stem}.results.json"),
        ]
        results_path = next((c for c in candidates_results if c.exists()), None)
        if results_path is None:
            gl = sorted(mp.parent.glob(f"{stem}*.results.jsonl"))
            if gl:
                results_path = gl[0]

        # bootstrap indices
        candidates_boot = [
            mp.with_name(f"{stem}.manifest.bootstrap_indices.npz"),
            mp.with_name(f"{stem}.bootstrap_indices.npz"),
            mp.with_name(f"{stem}.manifest_bootstrap_indices.npz"),
        ]
        boot_path = next((c for c in candidates_boot if c.exists()), None)
        if boot_path is None:
            gl = sorted(mp.parent.glob("*bootstrap_indices*.npz"))
            if gl:
                boot_path = gl[0]

        if results_path is None or boot_path is None:
            continue

        runs.append((task, model, n_splits, mp, results_path, boot_path))

    return runs


# ============================================================
# Collect per-setting metrics
# ============================================================
runs = find_runs(ABL_DIR)
print("Found runs:", len(runs))
if len(runs) == 0:
    sample_listing = sorted([str(p.relative_to(ABL_DIR)) for p in ABL_DIR.glob("**/*")][:200])
    raise RuntimeError(
        "No runs found under outputs/ablations/n_splits.\n"
        "Sample listing:\n" + "\n".join(sample_listing)
    )

records = []
for task, model, n_splits, manifest_path, results_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        continue

    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # Pull MAIN_SCORES explicitly so hidden_probe_oof can't disappear
    S = {}
    for k in MAIN_SCORES:
        arr = []
        for r in rows:
            v = r.get(k, np.nan)
            if v is None:
                v = np.nan
            arr.append(v)
        S[k] = np.asarray(arr, dtype=float)

    # load indices per score (prevents accidentally using the wrong array from NPZ)
    # fallback keeps legacy behavior if NPZ does not have these keys

    NPZ_KEY_MAP = {
        "lntp": "lntp",
        "mtp": "mtp",
        "egh_probe_oof": "egh",          # häufigster Key aus run_phase2 indices
        "hidden_probe_oof": "hidden",
    }

    for score_key, s_raw in S.items():
        boot_idx = load_bootstrap_indices(boot_path, key=NPZ_KEY_MAP.get(score_key))
        
        # Only hidden_probe_oof is expected to have missing values (dropped examples).
        if score_key != "hidden_probe_oof":
            if not np.isfinite(s_raw).all():
                # For non-hidden scores, missing values indicate a broken artifact -> skip.
                continue
            yy = y
            ss = s_raw
        else:
            mask = np.isfinite(s_raw)
            if mask.sum() < 10:
                continue
            yy = y[mask]
            ss = s_raw[mask]

        au, direction = auroc_with_best_direction(yy, ss)
        s = ss * direction

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(yy, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(yy, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "n_splits": int(n_splits),
            "score_key": score_key,
            "direction": float(direction),

            "N": int(len(yy)),
            "pos_rate": float(np.mean(yy)),

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
df = df.sort_values(["task", "model", "score_key", "n_splits"]).reset_index(drop=True)

df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)

if df.empty:
    raise RuntimeError("No rows computed. Check that results contain the expected MAIN_SCORES keys.")


# ============================================================
# Summary: mean/std across n_splits per task×model×score
# (This matches your ablation-table definition: AUROC + CI per setting;
#  and you can additionally report stability via std across settings.)
# ============================================================
g = df.groupby(["task", "model", "score_key"], as_index=False)
df_sum = g.agg(
    settings=("n_splits", "nunique"),
    auroc_mean_over_settings=("auroc_boot_mean", "mean"),
    auroc_std_over_settings=("auroc_boot_mean", "std"),
    spearman_mean_over_settings=("spearman_rho_boot_mean", "mean"),
    spearman_std_over_settings=("spearman_rho_boot_mean", "std"),
)

for c in ["auroc_std_over_settings", "spearman_std_over_settings"]:
    df_sum[c] = df_sum[c].fillna(0.0)

df_sum.to_csv(OUT_CSV_SUMMARY, index=False)
print("Wrote:", OUT_CSV_SUMMARY)


# ============================================================
# Plotting: overlay per task, lines = models, x = n_splits
# Separate panels per scorer to keep clarity (matches ablation intent)
# ============================================================
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])

score_order = [k for k in MAIN_SCORES if k in set(df["score_key"])]
score_order += sorted([k for k in set(df["score_key"]) if k not in score_order])

splits_order = sorted([int(x) for x in set(df["n_splits"]) if int(x) > 0])
if not splits_order:
    # fallback (if parsing failed)
    splits_order = sorted([int(x) for x in set(df["n_splits"])])

x = np.arange(len(splits_order), dtype=float)


def plot_overlay(metric: str, y_lim, title: str, outpath: Path):
    # grid: rows = scorers, cols = tasks
    fig, axes = plt.subplots(len(score_order), len(tasks),
                             figsize=(6.3 * len(tasks), 3.6 * len(score_order)),
                             sharey=False)
    if len(score_order) == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
    elif len(score_order) == 1:
        axes = np.array([axes])
    elif len(tasks) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, score_key in enumerate(score_order):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            for model in models:
                sub = df[(df["task"] == task) & (df["model"] == model) & (df["score_key"] == score_key)].copy()
                if sub.empty:
                    continue
                sub = sub.set_index("n_splits").reindex(splits_order).reset_index()

                if metric == "auroc":
                    yv = sub["auroc_boot_mean"].to_numpy(dtype=float)
                    lo = sub["auroc_ci95_lo"].to_numpy(dtype=float)
                    hi = sub["auroc_ci95_hi"].to_numpy(dtype=float)
                    hline = 0.5
                    ylabel = "AUROC"
                else:
                    yv = sub["spearman_rho_boot_mean"].to_numpy(dtype=float)
                    lo = sub["spearman_ci95_lo"].to_numpy(dtype=float)
                    hi = sub["spearman_ci95_hi"].to_numpy(dtype=float)
                    hline = 0.0
                    ylabel = "Spearman ρ"

                yerr = np.vstack([yv - lo, hi - yv])

                ax.plot(x, yv, marker="o", label=MODEL_PRETTY.get(model, model))
                ax.errorbar(x, yv, yerr=yerr, fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
                            elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)

            ax.axhline(hline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in splits_order])
            ax.set_ylim(*y_lim)
            ax.grid(False)

            if c == 0:
                ax.set_ylabel(f"{pretty_score(score_key)}\n{ylabel}")
            if r == 0:
                ax.set_title(TASK_PRETTY.get(task, task))

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),  # etwas nach unten verschoben
        ncol=len(models),
        frameon=False,
        title="Model",
    )

    fig.suptitle(title, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.93])  # mehr Platz oben reservieren
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)


plot_overlay(
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="OOF Robustness (n_splits) — AUROC ± 95% CI across CV fold counts",
    outpath=FIGS_DIR / "fig_ablation_n_splits_auroc_overlay.pdf",
)

plot_overlay(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="OOF Robustness (n_splits) — Spearman ρ ± 95% CI across CV fold counts",
    outpath=FIGS_DIR / "fig_ablation_n_splits_spearman_overlay.pdf",
)

print("Done. Figures in:", FIGS_DIR)