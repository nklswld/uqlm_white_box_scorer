# phase_2_medical/analysis/ablations/analyze_bootstrap_budget.py
#
# Ablation: Bootstrap Budget (B ∈ {1000, 2000, 5000, 10000})
# Goal: show CI-width convergence / numerical stability as B increases.
#
# Reads ablation artifacts from:
#   phase_2_medical/outputs/ablations/bootstrap_budget/...
#
# Expected per run:
#   *.manifest.json
#   corresponding *.results.jsonl
#   corresponding *.manifest.bootstrap_indices.npz  (or any *bootstrap_indices*.npz)
#
# Produces:
#   - outputs/ablations/bootstrap_budget/analysis_bootstrap_budget_metrics.csv
#   - outputs/ablations/bootstrap_budget/analysis_bootstrap_budget_summary.csv
#   - outputs/figs/ablations/bootstrap_budget/
#       fig_ablation_bootstrap_budget_ciwidth_auroc.pdf
#       fig_ablation_bootstrap_budget_ciwidth_spearman.pdf
#       fig_ablation_bootstrap_budget_delta_ci_auroc.pdf          (Option B)
#       fig_ablation_bootstrap_budget_delta_ci_spearman.pdf       (Option B)
#
# IMPORTANT:
#   - Styling matches phase2_figures.py and the other ablation analyzers.
#   - Option A implemented: tight, global y-axis based on observed range (same across panels).
#   - Option B implemented: Δ CI half-width relative to B=max (usually 10000).

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
FONT_SCALE = 1.35

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


def pretty_score(score_key: str) -> str:
    k = str(score_key).lower()
    return SCORE_PRETTY.get(k, k)


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical

ABL_DIR = ROOT / "outputs" / "ablations" / "bootstrap_budget"
FIGS_DIR = ROOT / "outputs" / "figs" / "ablations" / "bootstrap_budget"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
ABL_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FIGS_DIR / "analysis_bootstrap_budget_metrics.csv"
OUT_CSV_SUMMARY = FIGS_DIR / "analysis_bootstrap_budget_summary.csv"


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
# Helpers (same conventions as other ablation analyzers)
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
    # canonical (if ever used)
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

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
    if rhos.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi


# ============================================================
# Run discovery:
#   typical: .../bootstrap_budget/<task_model>/B_1000/*.manifest.json
# ============================================================
def parse_task_model_from_folders(manifest_path: Path):
    # .../bootstrap_budget/medqa_biomistral/B_1000/<file>.manifest.json
    try:
        task_model = manifest_path.parent.parent.name.lower()
        parts = task_model.split("_")
        task = parts[0]
        model = parts[1] if len(parts) > 1 else "unknown"
        return task, model
    except Exception:
        return "unknown", "unknown"


def parse_B(manifest_path: Path, manifest: dict):
    for k in ["B", "bootstrap_B", "bootstrap_budget"]:
        if k in manifest:
            try:
                return int(manifest[k])
            except Exception:
                pass

    folder = manifest_path.parent.name.lower()
    m = re.match(r"b[_\-]?(\d+)$", folder)
    if m:
        return int(m.group(1))

    name = manifest_path.name.lower()
    m = re.search(r"b(\d+)", name)
    if m:
        return int(m.group(1))

    txt = str(manifest_path).lower()
    m = re.search(r"b[_\-]?(\d+)", txt)
    if m:
        return int(m.group(1))

    return -1


def find_run_files(manifest_path: Path):
    stem = manifest_path.name.replace(".manifest.json", "")

    results_path = manifest_path.with_name(stem + ".results.jsonl")
    if not results_path.exists():
        alt = manifest_path.with_name(stem + ".results.json")
        results_path = alt if alt.exists() else None

    boot_path = manifest_path.with_name(stem + ".manifest.bootstrap_indices.npz")
    if not boot_path.exists():
        gl = sorted(manifest_path.parent.glob("*bootstrap_indices*.npz"))
        boot_path = gl[0] if gl else None

    return results_path, boot_path


runs = []
for mp in sorted(ABL_DIR.rglob("*.manifest.json")):
    try:
        manifest = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        continue

    task, model = parse_task_model_from_folders(mp)
    B = parse_B(mp, manifest)

    results_path, boot_path = find_run_files(mp)
    if results_path is None or boot_path is None:
        continue

    if not Path(results_path).exists() or not Path(boot_path).exists():
        continue

    runs.append((task, model, B, mp, Path(results_path), Path(boot_path)))

print("Found runs:", len(runs))
if len(runs) == 0:
    listing = sorted([str(p.relative_to(ABL_DIR)) for p in ABL_DIR.glob("**/*")][:250])
    raise RuntimeError("No runs found. Sample listing:\n" + "\n".join(listing))


# ============================================================
# Compute metrics + CI widths
# ============================================================
records = []
for task, model, B, manifest_path, results_path, boot_path in runs:
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

    keys = [k for k in keys if k in MAIN_SCORES]
    if not keys:
        continue

    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
    boot_idx = load_bootstrap_indices(boot_path)

    for score_key, s_raw in S.items():
        _, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "B": int(B),
            "score_key": score_key,
            "direction": float(direction),
            "N": int(len(y)),
            "pos_rate": float(y.mean()),

            "auroc_boot_mean": float(au_mean),
            "auroc_ci95_lo": float(au_lo),
            "auroc_ci95_hi": float(au_hi),
            "auroc_ci95_width": float(au_hi - au_lo),
            "auroc_ci95_halfwidth": float((au_hi - au_lo) / 2.0),

            "spearman_boot_mean": float(sp_mean),
            "spearman_ci95_lo": float(sp_lo),
            "spearman_ci95_hi": float(sp_hi),
            "spearman_ci95_width": float(sp_hi - sp_lo),
            "spearman_ci95_halfwidth": float((sp_hi - sp_lo) / 2.0),

            "manifest_file": str(manifest_path),
            "results_file": str(results_path),
            "boot_file": str(boot_path),
        })

df = pd.DataFrame(records)
df = df.sort_values(["task", "model", "score_key", "B"]).reset_index(drop=True)
df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)

if df.empty:
    raise RuntimeError("No metrics computed. Check that results contain MAIN_SCORES and run discovery is correct.")


# ============================================================
# Summary: aggregate CI half-width across tasks/models
# ============================================================
g = df.groupby(["score_key", "B"], as_index=False)
df_sum = g.agg(
    auroc_ci_halfwidth_mean=("auroc_ci95_halfwidth", "mean"),
    auroc_ci_halfwidth_median=("auroc_ci95_halfwidth", "median"),
    spearman_ci_halfwidth_mean=("spearman_ci95_halfwidth", "mean"),
    spearman_ci_halfwidth_median=("spearman_ci95_halfwidth", "median"),
    runs=("task", "count"),
)
df_sum = df_sum.sort_values(["score_key", "B"]).reset_index(drop=True)
df_sum.to_csv(OUT_CSV_SUMMARY, index=False)
print("Wrote:", OUT_CSV_SUMMARY)


# ============================================================
# Plotting setup
# ============================================================
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])

score_order = [k for k in SCORE_ORDER if k in set(df["score_key"])]
score_order += sorted([k for k in set(df["score_key"]) if k not in score_order])

Bs = sorted([int(x) for x in set(df["B"]) if int(x) > 0])
xpos = np.arange(len(Bs), dtype=float)


# ============================================================
# Option A: CI half-width vs B with tight global y-axis
# ============================================================
def plot_ciwidth(metric: str, title: str, outpath: Path):
    # choose column
    if metric == "auroc":
        col = "auroc_ci95_halfwidth"
        ylabel = "95% CI half-width (AUROC)"
    else:
        col = "spearman_ci95_halfwidth"
        ylabel = "95% CI half-width (Spearman ρ)"

    # -------- Option A: global tight y-limits (same across panels) --------
    all_vals = df[col].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        global_y_min, global_y_max = 0.0, 0.05
    else:
        pad = 0.001
        global_y_min = float(np.min(all_vals)) - pad
        global_y_max = float(np.max(all_vals)) + pad
        if global_y_max <= global_y_min:
            global_y_min, global_y_max = float(np.min(all_vals)) - 0.01, float(np.max(all_vals)) + 0.01
        global_y_min = max(0.0, global_y_min)

    # grid: rows=models, cols=tasks
    fig, axes = plt.subplots(
        len(models), len(tasks),
        figsize=(6.3 * len(tasks), 3.7 * len(models)),
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
            sub_tm = df[(df["task"] == task) & (df["model"] == model)].copy()
            if sub_tm.empty:
                continue

            for score_key in score_order:
                sub = sub_tm[sub_tm["score_key"] == score_key].copy()
                if sub.empty:
                    continue

                sub = sub.set_index("B").reindex(Bs).reset_index()
                yv = sub[col].to_numpy(dtype=float)

                ax.plot(xpos, yv, marker="o", label=pretty_score(score_key))

            ax.set_xticks(xpos)
            ax.set_xticklabels([str(b) for b in Bs])
            ax.set_ylim(global_y_min, global_y_max)

            if r == len(models) - 1:
                ax.set_xlabel("Bootstrap budget B")
            if c == 0:
                ax.set_ylabel(ylabel)

            if r == 0:
                ax.set_title(TASK_PRETTY.get(task, task))
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(4, len(score_order)),
        frameon=False,
        title="Scorer"
    )
    fig.suptitle(title, y=0.993)
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    fig.subplots_adjust(hspace=0.42)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


# ============================================================
# Option B: Δ CI half-width vs B=max (typically 10000)
# ============================================================
def plot_delta_ci(metric: str, title: str, outpath: Path):
    if metric == "auroc":
        col = "auroc_ci95_halfwidth"
        ylabel = "Δ 95% CI half-width\n(AUROC) (vs B=max)"
    else:
        col = "spearman_ci95_halfwidth"
        ylabel = "Δ 95% CI half-width\n(Spearman ρ) (vs B=max)"

    B_ref = max(Bs) if Bs else None
    if B_ref is None:
        raise RuntimeError("No bootstrap budgets found (Bs empty).")

    # compute delta column (per task-model-score): y(B) - y(B_ref)
    df_delta = df[["task", "model", "score_key", "B", col]].copy()
    ref = df_delta[df_delta["B"] == B_ref][["task", "model", "score_key", col]].rename(columns={col: "ref"})
    df_delta = df_delta.merge(ref, on=["task", "model", "score_key"], how="left")
    df_delta["delta"] = df_delta[col] - df_delta["ref"]

    # tight global y-limits around deltas (same across panels)
    all_vals = df_delta["delta"].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        global_y_min, global_y_max = -0.01, 0.01
    else:
        pad = 0.0005
        global_y_min = float(np.min(all_vals)) - pad
        global_y_max = float(np.max(all_vals)) + pad
        if global_y_max <= global_y_min:
            global_y_min, global_y_max = float(np.min(all_vals)) - 0.01, float(np.max(all_vals)) + 0.01

    fig, axes = plt.subplots(
        len(models), len(tasks),
        figsize=(6.3 * len(tasks), 3.7 * len(models)),
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
            sub_tm = df_delta[(df_delta["task"] == task) & (df_delta["model"] == model)].copy()
            if sub_tm.empty:
                continue

            for score_key in score_order:
                sub = sub_tm[sub_tm["score_key"] == score_key].copy()
                if sub.empty:
                    continue
                sub = sub.set_index("B").reindex(Bs).reset_index()
                yv = sub["delta"].to_numpy(dtype=float)
                ax.plot(xpos, yv, marker="o", label=pretty_score(score_key))

            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.set_xticks(xpos)
            ax.set_xticklabels([str(b) for b in Bs])
            ax.set_ylim(global_y_min, global_y_max)

            if r == len(models) - 1:
                ax.set_xlabel("Bootstrap budget B")
            if c == 0:
                ax.set_ylabel(ylabel)

            if r == 0:
                ax.set_title(TASK_PRETTY.get(task, task))
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(4, len(score_order)),
        frameon=False,
        title="Scorer"
    )
    fig.suptitle(title, y=0.993)
    fig.tight_layout(rect=[0, 0, 1, 0.952])
    fig.subplots_adjust(hspace=0.42)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


# ============================================================
# Generate figures
# ============================================================
plot_ciwidth(
    metric="auroc",
    title="Bootstrap Budget — 95% CI half-width vs B (AUROC)",
    outpath=FIGS_DIR / "fig_ablation_bootstrap_budget_ciwidth_auroc.pdf",
)
plot_ciwidth(
    metric="spearman",
    title="Bootstrap Budget — 95% CI half-width vs B (Spearman ρ)",
    outpath=FIGS_DIR / "fig_ablation_bootstrap_budget_ciwidth_spearman.pdf",
)

# Option B: Δ plots vs max B
plot_delta_ci(
    metric="auroc",
    title="Bootstrap Budget — Δ 95% CI half-width vs B=max (AUROC)",
    outpath=FIGS_DIR / "fig_ablation_bootstrap_budget_delta_ci_auroc.pdf",
)
plot_delta_ci(
    metric="spearman",
    title="Bootstrap Budget — Δ 95% CI half-width vs B=max (Spearman ρ)",
    outpath=FIGS_DIR / "fig_ablation_bootstrap_budget_delta_ci_spearman.pdf",
)

print("[OK] Bootstrap budget ablation done. Outputs in:", FIGS_DIR)