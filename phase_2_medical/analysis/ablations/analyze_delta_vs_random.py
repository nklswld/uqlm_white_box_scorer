# phase_2_medical/analysis/ablations/analyze_delta_vs_random.py
#
# Ablation: Δ vs Random Baseline (paired)
# Uses the *general* Phase-2 outputs in: phase_2_medical/outputs/final/
#
# What it does:
#   For each Task × Model × Scorer, compute bootstrap distributions of
#     - AUROC
#     - Spearman ρ
#   using the saved bootstrap indices from the corresponding
#   *.manifest.bootstrap_indices.npz file.
#
# Then compute paired deltas vs random baselines:
#   ΔAUROC    = AUROC_boot - 0.5
#   ΔSpearman = rho_boot   - 0.0
#
# Outputs:
#   CSVs:
#     - phase_2_medical/outputs/ablations/delta_vs_random/analysis_delta_vs_random_metrics.csv
#     - phase_2_medical/outputs/ablations/delta_vs_random/analysis_delta_vs_random_summary.csv
#   Figures:
#     - phase_2_medical/outputs/figures_tables/ablations/delta_vs_random/fig_ablation_delta_vs_random_auroc.pdf
#     - phase_2_medical/outputs/figures_tables/ablations/delta_vs_random/fig_ablation_delta_vs_random_spearman.pdf
#
# Style/structure: aligned with phase2_figures.py and the other ablation analyzers.

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
FONT_SCALE = 1.35

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

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,

    "axes.titlepad": 12,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

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

SCORE_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]
MAIN_SCORES = set(SCORE_ORDER)

BASELINES = {"auroc": 0.5, "spearman": 0.0}

SCORE_TO_BOOTKEY = {
    "lntp": "lntp",
    "mtp": "mtp",
    "egh_probe_oof": "egh",
    "hidden_probe_oof": "hidden",
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
                              fmt="{:.3f}", fontsize=None, pad_frac=0.02):
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


def load_bootstrap_indices_map(boot_path: Path) -> dict:
    """
    Load ALL bootstrap index arrays from the NPZ as a dict.
    Each key corresponds to a scorer (e.g., lntp, mtp, egh, hidden) and
    optionally includes hidden_kept_indices.
    """
    z = np.load(boot_path, allow_pickle=True)
    out = {}
    for k in z.files:
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack(arr, axis=0)
        out[str(k).lower()] = arr.astype(int)
    return out


def find_label_key(example: dict):
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")


def extract_scores(example: dict):
    """
    Extract score fields robustly from Phase-2 results rows.

    IMPORTANT:
    - Phase-2 results.jsonl stores scores as top-level keys (lntp, mtp, egh_probe_oof, hidden_probe_oof).
    - hidden_probe_oof can be null -> must be preserved as np.nan so downstream can subset via hidden_kept_indices.
    """
    scores = {}

    # 1) Prefer explicit main scorer keys if present (even if None).
    for k in MAIN_SCORES:
        if k in example:
            v = example.get(k, None)
            scores[str(k).lower()] = (np.nan if v is None else float(v))

    # 2) Also support nested dict schemas (if any run emits them)
    if "scores" in example and isinstance(example["scores"], dict):
        for k, v in example["scores"].items():
            if v is None:
                scores[str(k).lower()] = np.nan
            elif isinstance(v, (float, int)):
                scores[str(k).lower()] = float(v)

    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        for k, v in example["wb_scores"].items():
            if v is None:
                scores[str(k).lower()] = np.nan
            elif isinstance(v, (float, int)):
                scores[str(k).lower()] = float(v)

    # 3) Fallback heuristic (kept, but now it won't drop explicit keys)
    for k, v in example.items():
        kk = str(k).lower()
        if kk in scores:
            continue
        if isinstance(v, (float, int)):
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)

    return scores


def auroc_best_direction(y: np.ndarray, s: np.ndarray):
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


def spearman_rho(s: np.ndarray, y: np.ndarray) -> float:
    # Spearman via pandas for robust ranking/ties handling
    rho = pd.Series(s).corr(pd.Series(y), method="spearman")
    return float(rho) if pd.notna(rho) else np.nan


def bootstrap_metric_distributions(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray):
    """Return (auroc_dist, spearman_dist) computed on each bootstrap sample."""
    aucs = []
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if yy.min() == yy.max():
            # AUROC undefined if only one class in sample
            continue
        aucs.append(roc_auc_score(yy, ss))
        rr = spearman_rho(ss, yy)
        if not np.isnan(rr):
            rhos.append(rr)

    aucs = np.asarray(aucs, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    return aucs, rhos


def ci_from_dist(dist: np.ndarray, alpha=0.05):
    if dist.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(dist))
    lo = float(np.quantile(dist, alpha / 2))
    hi = float(np.quantile(dist, 1 - alpha / 2))
    return mean, lo, hi


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
FINAL_DIR = ROOT / "outputs" / "final"

FIGS_OUT = ROOT / "outputs" / "figures_tables" / "ablations" / "delta_vs_random"
FIGS_OUT.mkdir(parents=True, exist_ok=True)

# CSVs sollen ebenfalls hier gespeichert werden
CSV_OUT = FIGS_OUT


print("ROOT =", ROOT)
print("FINAL_DIR =", FINAL_DIR)
print("FINAL_DIR exists =", FINAL_DIR.exists())
print("ABL_OUT =", FIGS_OUT)
print("FIGS_OUT =", FIGS_OUT)

if not FINAL_DIR.exists():
    raise FileNotFoundError(f"Final folder not found: {FINAL_DIR}")


# ============================================================
# Discover final runs
# Expect files:
#   <task>_<model>.B5000.results.jsonl
#   <task>_<model>.B5000.manifest.bootstrap_indices.npz
# ============================================================
runs = []
for results_path in sorted(FINAL_DIR.glob("*.results.jsonl")):
    name = results_path.name  # e.g. medqa_mistral.B5000.results.jsonl
    if ".B" not in name:
        continue

    # parse task/model from prefix "<task>_<model>"
    prefix = name.split(".B")[0]
    if "_" not in prefix:
        continue
    task, model = prefix.split("_", 1)
    task = task.lower()
    model = model.lower()

    # bootstrap indices: try canonical naming and fallback glob
    stem = name.replace(".results.jsonl", "")
    boot_path = FINAL_DIR / f"{stem}.manifest.bootstrap_indices.npz"
    if not boot_path.exists():
        gl = sorted(FINAL_DIR.glob(f"{stem}*bootstrap_indices*.npz"))
        boot_path = gl[0] if gl else None

    if boot_path is None or not boot_path.exists():
        print("[WARN] Missing bootstrap indices for:", results_path.name)
        continue

    runs.append((task, model, results_path, boot_path))

print("Found final runs:", len(runs))
if len(runs) == 0:
    listing = sorted([str(p.relative_to(FINAL_DIR)) for p in FINAL_DIR.glob("*")][:200])
    raise RuntimeError("No runs found in final/. Sample listing:\n" + "\n".join(listing))


# ============================================================
# Compute paired deltas vs random baselines (AUROC, Spearman)
# ============================================================
records = []

for task, model, results_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        continue

    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted([str(k).lower() for k in keys if str(k).lower() in MAIN_SCORES])

    if not keys:
        print("[WARN] No MAIN_SCORES found in:", results_path.name)
        continue

    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}

    boot_map = load_bootstrap_indices_map(boot_path)

    # optional: hidden kept indices (global indices into full dataset)
    hidden_kept = boot_map.get("hidden_kept_indices", None)

    for score_key, s_raw in S.items():
        score_l = score_key.lower()   # <-- FIX: define early
        s_raw = np.asarray(s_raw, dtype=float)

        # For hidden: direction should be determined on the kept subset (same population as bootstrap)
        if score_l == "hidden_probe_oof" and ("hidden_kept_indices" in boot_map):
            hk = boot_map["hidden_kept_indices"]
            au_full, direction = auroc_best_direction(y[hk], s_raw[hk])
        else:
            au_full, direction = auroc_best_direction(y, s_raw)

        s = s_raw * direction

        # pick correct bootstrap indices for this score
        boot_key = SCORE_TO_BOOTKEY.get(score_l, score_l)

        if boot_key not in boot_map:
            print(f"[WARN] No bootstrap indices for score_key={score_key} in {boot_path.name}; skipping.")
            continue

        boot_idx = boot_map[boot_key]

        # Special case: hidden is bootstrapped on the kept-subset (indices are relative to that subset)
        if score_l == "hidden_probe_oof":
            if hidden_kept is None:
                raise KeyError(
                    f"hidden_kept_indices missing in {boot_path.name} but required for hidden bootstrap."
                )
            # hidden_kept are indices into FULL y/s arrays
            y_use = y[hidden_kept]
            s_use = s[hidden_kept]
        else:
            y_use = y
            s_use = s

        au_dist, sp_dist = bootstrap_metric_distributions(y_use, s_use, boot_idx)

        # paired deltas vs random
        d_au = au_dist - BASELINES["auroc"]
        d_sp = sp_dist - BASELINES["spearman"]

        # CI in delta-space (paired)
        d_au_mean, d_au_lo, d_au_hi = ci_from_dist(d_au, alpha=0.05)
        d_sp_mean, d_sp_lo, d_sp_hi = ci_from_dist(d_sp, alpha=0.05)

        # also store absolute metrics for reference
        au_mean, au_lo, au_hi = ci_from_dist(au_dist, alpha=0.05)
        sp_mean, sp_lo, sp_hi = ci_from_dist(sp_dist, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "score_key": score_key,
            "direction": float(direction),
            "N": int(len(y)),
            "pos_rate": float(y.mean()),

            "auroc_full": float(au_full),
            "auroc_boot_mean": float(au_mean),
            "auroc_ci95_lo": float(au_lo),
            "auroc_ci95_hi": float(au_hi),

            "delta_auroc_boot_mean": float(d_au_mean),
            "delta_auroc_ci95_lo": float(d_au_lo),
            "delta_auroc_ci95_hi": float(d_au_hi),
            "delta_auroc_sig_pos": bool(np.isfinite(d_au_lo) and d_au_lo > 0.0),

            "spearman_boot_mean": float(sp_mean),
            "spearman_ci95_lo": float(sp_lo),
            "spearman_ci95_hi": float(sp_hi),

            "delta_spearman_boot_mean": float(d_sp_mean),
            "delta_spearman_ci95_lo": float(d_sp_lo),
            "delta_spearman_ci95_hi": float(d_sp_hi),
            "delta_spearman_sig_pos": bool(np.isfinite(d_sp_lo) and d_sp_lo > 0.0),

            "results_file": str(results_path),
            "boot_file": str(boot_path),
        })

df = pd.DataFrame(records)
df = df.sort_values(["task", "model", "score_key"]).reset_index(drop=True)

out_metrics = CSV_OUT / "analysis_delta_vs_random_metrics.csv"
df.to_csv(out_metrics, index=False)
print("Wrote:", out_metrics)



# Summary (compact)
df_sum = df[[
    "task", "model", "score_key",
    "delta_auroc_boot_mean", "delta_auroc_ci95_lo", "delta_auroc_ci95_hi", "delta_auroc_sig_pos",
    "delta_spearman_boot_mean", "delta_spearman_ci95_lo", "delta_spearman_ci95_hi", "delta_spearman_sig_pos",
]].copy()

out_summary = CSV_OUT / "analysis_delta_vs_random_summary.csv"
df_sum.to_csv(out_summary, index=False)
print("Wrote:", out_summary)


# ============================================================
# Plotting: compact grid (rows=models, cols=tasks)
# One figure per metric: ΔAUROC and ΔSpearman
# ============================================================
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])

score_order = [k for k in SCORE_ORDER if k in set(df["score_key"])]
score_order += sorted([k for k in set(df["score_key"]) if k not in score_order])


def plot_delta(metric: str, outpath: Path):
    if metric == "auroc":
        mean_col = "delta_auroc_boot_mean"
        lo_col = "delta_auroc_ci95_lo"
        hi_col = "delta_auroc_ci95_hi"
        ylabel = r"$\Delta$AUROC (vs. 0.5)"
        title = r"$\Delta$ vs Random Baseline (paired) — $\Delta$AUROC $\pm$ 95% CI"
    else:
        mean_col = "delta_spearman_boot_mean"
        lo_col = "delta_spearman_ci95_lo"
        hi_col = "delta_spearman_ci95_hi"
        ylabel = r"$\Delta$ Spearman $\rho$ (vs. 0.0)"
        title = r"$\Delta$ vs Random Baseline (paired) — $\Delta\rho$ $\pm$ 95% CI"

    # global y-limits (tight, symmetric-ish around 0)
    vals = df[[lo_col, hi_col]].to_numpy(dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        y_min, y_max = -0.05, 0.20
    else:
        pad = 0.01 * (float(np.max(vals)) - float(np.min(vals)) + 1e-9)
        y_min = float(np.min(vals)) - pad
        y_max = float(np.max(vals)) + pad
        # ensure 0 line visible
        y_min = min(y_min, -0.01)
        y_max = max(y_max, 0.01)
        # give headroom so value labels don't collide with subplot titles (without blowing up the scale)
        y_max = y_max + 0.10 * (y_max - y_min + 1e-9)


    fig, axes = plt.subplots(
        len(models), len(tasks),
        figsize=(5.6 * len(tasks), 3.9 * len(models)),
        sharey=True
    )

    if len(models) == 1 and len(tasks) == 1:
        axes = np.array([[axes]])
    elif len(models) == 1:
        axes = np.array([axes])
    elif len(tasks) == 1:
        axes = np.array([[ax] for ax in axes])

    for r, model in enumerate(models):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            sub = df[(df["task"] == task) & (df["model"] == model)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            sub["score_key"] = pd.Categorical(sub["score_key"], categories=score_order, ordered=True)
            sub = sub.sort_values("score_key")

            x = np.arange(len(score_order), dtype=float)

            y = sub.set_index("score_key").reindex(score_order)[mean_col].to_numpy(dtype=float)
            lo = sub.set_index("score_key").reindex(score_order)[lo_col].to_numpy(dtype=float)
            hi = sub.set_index("score_key").reindex(score_order)[hi_col].to_numpy(dtype=float)

            yerr_low = y - lo
            yerr_high = hi - y

            ax.bar(x, y, width=0.65)
            ax.errorbar(
                x, y, yerr=[yerr_low, yerr_high],
                fmt="none",
                capsize=ERRORBAR_CAPSIZE,
                ecolor="black",
                elinewidth=ERRORBAR_LINEWIDTH,
                capthick=ERRORBAR_CAPTHICK,
            )
            ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)

            ax.set_xticks(x)
            ax.set_xticklabels([pretty_score(k) for k in score_order], rotation=20, ha="right")

            ax.set_ylim(y_min, y_max)
            if c == 0:
                ax.set_ylabel(ylabel)

            if r == 0:
                ax.set_title(TASK_PRETTY.get(task, task), pad=13) 

            if c == len(tasks) - 1:
                ax.text(
                    1.02, 0.5,
                    MODEL_PRETTY.get(model, model),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="left",
                    fontsize=mpl.rcParams["axes.labelsize"]
                )

            add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}", pad_frac=0.02)

    fig.suptitle(
        title,
        y=0.995,
        fontsize=mpl.rcParams["figure.titlesize"] * 1.15
    )

    fig.subplots_adjust(
        top=0.86,     
        bottom=0.10,
        left=0.08,
        right=0.96,
        hspace=0.28,
        wspace=0.18
    )

    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


plot_delta("auroc", FIGS_OUT / "fig_ablation_delta_vs_random_auroc.pdf")
plot_delta("spearman", FIGS_OUT / "fig_ablation_delta_vs_random_spearman.pdf")

print("[OK] Δ vs random baseline (paired) done.")
print("CSVs in:", CSV_OUT)
print("Figs in:", FIGS_OUT)