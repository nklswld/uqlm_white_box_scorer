"""
Phase 2 figure/metric generator for medical QA evaluation runs.

Reads per-run artifacts under outputs/final/: *.manifest.json, matching *.results.jsonl, and
*.manifest.bootstrap_indices.npz (precomputed resample indices).
Writes publication-ready PDF figures into outputs/figures_tables/figures_general (including grouped and story panels).
Outputs: AUROC point estimates with 95% percentile bootstrap CIs and Spearman ρ (bootstrap mean + 95% CI).
Determinism: no RNG is used; all bootstrap resamples are driven by precomputed index arrays in the input NPZ files.
"""

# phase_2_medical/analysis/phase2_figures.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------
FONT_SCALE = 1.5

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size":              int(12   * FONT_SCALE),
    "axes.titlesize":         int(13   * FONT_SCALE),
    "axes.labelsize":         int(12   * FONT_SCALE),
    "xtick.labelsize":        int(12   * FONT_SCALE),
    "ytick.labelsize":        int(11   * FONT_SCALE),
    "legend.fontsize":        int(11   * FONT_SCALE),
    "legend.title_fontsize":  int(11   * FONT_SCALE),
    "figure.titlesize":       int(14.5 * FONT_SCALE),
    "axes.titlepad": 12,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.major.size":  4.5, "ytick.major.size":  4.5,
    "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
    "xtick.minor.size":  3.0, "ytick.minor.size":  3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT  = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final"
FIGS  = ROOT / "outputs" / "figures_tables" / "figures_general"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MAIN_SCORES    = {"lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"}
AUROC_YLIM     = (0.45, 0.80)
SPEARMAN_YLIM  = (-0.05, 0.60)

ERRORBAR_CAPSIZE    = 4
ERRORBAR_LINEWIDTH  = 1.6
ERRORBAR_CAPTHICK   = 1.6
BASELINE_LINEWIDTH  = 1.4

SCORE_PRETTY = {"lntp": "LNTP", "mtp": "MTP", "egh_probe_oof": "EGH", "hidden_probe_oof": "Hidden"}
SCORE_ORDER  = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}
MODEL_COLOR  = {"mistral": "tab:blue", "biomistral": "tab:orange"}
MODEL_ORDER  = ["mistral", "biomistral"]

SCORE_TO_BOOTKEY = {
    "lntp": "lntp", "mtp": "mtp",
    "egh_probe_oof": "egh", "egh_probe_ge": "egh_ge",
    "egh_probe_g_only": "egh_g", "egh_probe_e_only": "egh_e",
    "egh_probe_scalar_only": "egh_scalar",
    "hidden_probe_oof": "hidden",
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_savefig(fig, outpath: Path, **kwargs):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs); return outpath
    except PermissionError:
        for k in range(2, 50):
            alt = outpath.with_name(f"{outpath.stem}_v{k}{outpath.suffix}")
            try:
                fig.savefig(alt, **kwargs)
                print(f"[WARN] Locked → wrote {alt.name}"); return alt
            except PermissionError:
                continue
        raise


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_bootstrap_map(npz_path: Path):
    z, out = np.load(npz_path, allow_pickle=True), {}
    for k in z.files:
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack(arr, axis=0)
        if isinstance(arr, np.ndarray):
            out[str(k).lower()] = arr.astype(int)
    for alias in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if alias in out and "indices" not in out:
            out["indices"] = out[alias]
    if "indices" not in out and len(out) == 1:
        out["indices"] = next(iter(out.values()))
    return out


def find_label_key(example: dict):
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example: return k
    raise KeyError("No label key found.")


def extract_scores(example: dict):
    for key in ("scores", "wb_scores"):
        if key in example and isinstance(example[key], dict):
            return example[key]
    return {str(k).lower(): float(v) for k, v in example.items()
            if isinstance(v, (float, int)) and any(s in str(k).lower() for s in ["lntp", "mtp", "egh", "hidden"])}


def auroc_with_best_direction(y, s):
    au = roc_auc_score(y, s)
    return (roc_auc_score(y, -s), -1.0) if au < 0.5 else (au, +1.0)


def bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05):
    aucs = [roc_auc_score(y[i], s[i]) for i in boot_idx if y[i].min() != y[i].max()]
    if not aucs: return np.nan, np.nan, np.nan
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.quantile(aucs, alpha/2)), float(np.quantile(aucs, 1-alpha/2))


def bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05):
    rhos = [pd.Series(s[i]).corr(pd.Series(y[i]), method="spearman")
            for i in boot_idx if y[i].min() != y[i].max()]
    rhos = [r for r in rhos if not pd.isna(r)]
    if not rhos: return np.nan, np.nan, np.nan
    rhos = np.array(rhos)
    return float(rhos.mean()), float(np.quantile(rhos, alpha/2)), float(np.quantile(rhos, 1-alpha/2))


def infer_task_model_from_manifest(path: Path):
    m = json.loads(path.read_text(encoding="utf-8"))
    model_name = str(m.get("config", {}).get("model_name", "")).lower()
    return str(m.get("task", "")).lower(), "biomistral" if "bio" in model_name else "mistral"


def pretty_score(k: str):
    return SCORE_PRETTY.get(str(k).lower(), k)


def model_legend_handles():
    return [Line2D([0], [0], color=MODEL_COLOR[m], marker="o", linestyle="-",
                   linewidth=2.0, markersize=6.0, label=MODEL_PRETTY[m])
            for m in MODEL_ORDER]


def add_model_legend(fig, ncol=2, y=0.925):
    fig.legend(handles=model_legend_handles(), title="Model", loc="upper center",
               bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               handlelength=2.0, columnspacing=1.4)


def add_value_labels_above_ci(ax, xs, ys, yerr_high, fmt="{:.3f}", fontsize=None, pad_frac=0.0135):
    if fontsize is None: fontsize = VALUE_LABEL_FONTSIZE
    pad = pad_frac * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for x, y, eh in zip(xs, ys, yerr_high):
        if y is None or (isinstance(y, float) and np.isnan(y)): continue
        ax.text(float(x), float(y) + float(eh or 0) + pad, fmt.format(float(y)),
                ha="center", va="bottom", fontsize=fontsize)


def _errorbar(ax, x, y, yerr_low, yerr_high):
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none",
                capsize=ERRORBAR_CAPSIZE, ecolor="black",
                elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK)

# ---------------------------------------------------------------------
# Collect FINAL runs
# ---------------------------------------------------------------------
print("ROOT =", ROOT, "| FINAL exists =", FINAL.exists())
if not FINAL.exists():
    raise FileNotFoundError(f"FINAL directory not found: {FINAL}")

runs = []
for manifest_path in sorted(FINAL.glob("*.manifest.json")):
    results_path = manifest_path.with_suffix("").with_suffix(".results.jsonl")
    boot_path    = manifest_path.with_suffix("").with_suffix(".manifest.bootstrap_indices.npz")
    if not results_path.exists():
        print("[WARN] Missing results for", manifest_path.name); continue
    if not boot_path.exists():
        print("[WARN] Missing bootstrap npz for", manifest_path.name); continue
    task, model = infer_task_model_from_manifest(manifest_path)
    runs.append((task, model, results_path, manifest_path, boot_path))

print("Found FINAL runs:", [(t, m, p.name) for t, m, p, _, _ in runs])
if not runs:
    raise RuntimeError("No runs found. Check naming: *.manifest.json + *.results.jsonl + *.manifest.bootstrap_indices.npz")

# ---------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------
records, spearman_records, spearman_ci_rows = [], [], []

for task, model, results_path, manifest_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows: continue

    y_key      = find_label_key(rows[0])
    y          = np.array([int(r[y_key]) for r in rows], dtype=int)
    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]

    keys = set(score_dicts[0])
    for d in score_dicts[1:]: keys &= set(d)
    S = {k: np.array([d[k] for d in score_dicts], dtype=float)
         for k in sorted(keys) if k in MAIN_SCORES}

    boot_map    = load_bootstrap_map(boot_path)
    hidden_kept = boot_map.get("hidden_kept_indices", None)

    for score_name, s_raw in S.items():
        score_l  = score_name.lower()
        boot_key = SCORE_TO_BOOTKEY.get(score_l, score_l)
        boot_idx = boot_map.get(boot_key)
        if boot_idx is None:
            boot_idx = boot_map.get("indices")

        if score_l == "hidden_probe_oof" and hidden_kept is not None:
            au, direction = auroc_with_best_direction(y[hidden_kept], s_raw[hidden_kept])
        else:
            au, direction = auroc_with_best_direction(y, s_raw)

        s = s_raw * direction

        if score_l == "hidden_probe_oof":
            if hidden_kept is None:
                m = np.isfinite(s); y_use, s_use = y[m], s[m]
            else:
                y_use, s_use = y[hidden_kept], s[hidden_kept]
        else:
            y_use, s_use = y, s

        if boot_idx.shape[1] != len(y_use):
            raise ValueError(f"Bootstrap shape mismatch for {score_l}: {boot_idx.shape} vs N={len(y_use)}")

        mean_b, lo, hi     = bootstrap_ci_from_indices(y_use, s_use, boot_idx)
        rho                = pd.Series(s_use).corr(pd.Series(y_use), method="spearman")
        m_rho, lo_rho, hi_rho = bootstrap_spearman_ci_from_indices(y_use, s_use, boot_idx)

        base = dict(task=task, model=model, score=score_l)
        spearman_records.append({**base, "spearman_rho": float(rho), "direction": float(direction),
                                  "N": len(y_use), "pos_rate": float(np.mean(y_use))})
        spearman_ci_rows.append({**base, "spearman_rho_boot_mean": m_rho, "ci95_lo": lo_rho, "ci95_hi": hi_rho})
        records.append({**base, "direction": float(direction), "N": len(y_use),
                        "pos_rate": float(np.mean(y_use)), "auroc": float(au),
                        "auroc_boot_mean": float(mean_b), "ci95_lo": float(lo), "ci95_hi": float(hi),
                        "manifest_file": str(manifest_path), "results_file": str(results_path),
                        "boot_file": str(boot_path)})

df       = pd.DataFrame(records).sort_values(["task", "model", "auroc"], ascending=[True, True, False])
df_main  = df[df["score"].isin(MAIN_SCORES)].copy()

df_spear_main = pd.DataFrame(spearman_records).merge(
    pd.DataFrame(spearman_ci_rows), on=["task", "model", "score"], how="left")

# ---------------------------------------------------------------------
# Shared bar-plot builder (used by both AUROC and Spearman _bar functions)
# ---------------------------------------------------------------------
def _plot_bar(df_task, title, outpath,
              y_col, ylim, ylabel, baseline,
              sort_col=None):
    """Generic sorted bar chart with CI errorbars, value labels, and model legend."""
    dfp = df_task.copy()
    for c in ("task", "model", "score"):
        dfp[c] = dfp[c].astype(str).str.lower()

    multi_task = dfp["task"].nunique() > 1
    dfp["label"] = (dfp["task"].str.upper() + " | " + dfp["score"].map(pretty_score)
                    if multi_task else dfp["score"].map(pretty_score))
    dfp = dfp.sort_values(sort_col or y_col, ascending=False).reset_index(drop=True)

    x  = np.arange(len(dfp), dtype=float)
    yv = dfp[y_col].to_numpy(dtype=float)
    lo = dfp["ci95_lo"].to_numpy(dtype=float)
    hi = dfp["ci95_hi"].to_numpy(dtype=float)
    yerr_low, yerr_high = yv - lo, hi - yv

    n_bars = len(dfp)
    fig, ax = plt.subplots(figsize=(max(9.5, n_bars * 0.78), 7.5))

    ax.bar(x, yv, width=0.65, color=[MODEL_COLOR.get(m, "tab:gray") for m in dfp["model"]])
    _errorbar(ax, x, yv, yerr_low, yerr_high)
    ax.axhline(baseline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_size(int(10.5 * FONT_SCALE))

    fig.suptitle(title, y=0.975)
    add_model_legend(fig, ncol=2, y=0.915)

    ax.set_xticks(x)
    ax.set_xticklabels(dfp["label"].tolist(),
                       rotation=35 if multi_task else 0,
                       ha="right" if multi_task else "center")
    ax.tick_params(axis="x", labelsize=int(10 * FONT_SCALE))

    add_value_labels_above_ci(ax, x, yv, yerr_high)
    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)


def plot_auroc_bar(df_task, title, outpath):
    _plot_bar(df_task, title, outpath,
              y_col="auroc_boot_mean", ylim=AUROC_YLIM,
              ylabel="AUROC\n(bootstrap mean)", baseline=0.5,
              sort_col="auroc_boot_mean")


def plot_spearman_bar(df_task, title, outpath):
    _plot_bar(df_task, title, outpath,
              y_col="spearman_rho_boot_mean", ylim=SPEARMAN_YLIM,
              ylabel="Spearman ρ\n(bootstrap mean)", baseline=0.0)


# ---------------------------------------------------------------------
# Bar plots (single-task + ALL)
# ---------------------------------------------------------------------
df_main_filtered = df_main[df_main["score"].isin(MAIN_SCORES)].copy()
for task in df_main_filtered["task"].unique():
    plot_auroc_bar(df_main_filtered[df_main_filtered["task"] == task],
                  f"Phase 2 AUROC + 95% CI — {task}",
                  FIGS / f"fig_phase2_auroc_bar_{task}.pdf")
plot_auroc_bar(df_main_filtered,
               "Phase 2 AUROC + 95% CI — all runs (MedQA + PubMedQA shown together)",
               FIGS / "fig_phase2_auroc_bar_ALL.pdf")

df_spear_filtered = df_spear_main[df_spear_main["score"].isin(MAIN_SCORES)].copy()
for task in df_spear_filtered["task"].unique():
    plot_spearman_bar(df_spear_filtered[df_spear_filtered["task"] == task],
                     f"Phase 2 Spearman ρ + 95% CI — {task}",
                     FIGS / f"fig_phase2_spearman_bar_{task}.pdf")
plot_spearman_bar(df_spear_filtered,
                  "Phase 2 Spearman ρ + 95% CI — all runs (MedQA + PubMedQA shown together)",
                  FIGS / "fig_phase2_spearman_bar_ALL.pdf")

# ---------------------------------------------------------------------
# Shared grouped bar builder
# ---------------------------------------------------------------------
def _plot_grouped(df_task, title, outpath,
                  y_col, ylim, ylabel, baseline, sort_col=None):
    """Generic grouped bar chart (scorer × model) with CI errorbars and value labels."""
    dfp = df_task.copy()
    dfp["score"] = dfp["score"].str.lower()
    dfp["model"] = dfp["model"].str.lower()
    dfp = dfp[dfp["score"].isin(SCORE_ORDER)].copy()

    models = [m for m in MODEL_ORDER if m in set(dfp["model"])]
    width  = 0.38 if len(models) == 2 else 0.6
    x_base = np.arange(len(SCORE_ORDER), dtype=float)

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        xs, ys, ylo, yhi = x_base + offset, [], [], []
        for s in SCORE_ORDER:
            sub = dfp[(dfp["score"] == s) & (dfp["model"] == model)]
            if sub.empty:
                ys.append(np.nan); ylo.append(0.0); yhi.append(0.0)
            else:
                r = sub.sort_values(sort_col or y_col, ascending=False).iloc[0]
                yv = float(r[y_col])
                ys.append(yv)
                ylo.append(yv - float(r["ci95_lo"]))
                yhi.append(float(r["ci95_hi"]) - yv)
        ax.bar(xs, ys, width=width * 0.95, color=MODEL_COLOR.get(model, "tab:gray"))
        _errorbar(ax, xs, ys, ylo, yhi)
        add_value_labels_above_ci(ax, xs, ys, yhi, pad_frac=0.010)

    ax.axhline(baseline, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x_base)
    ax.set_xticklabels([pretty_score(s) for s in SCORE_ORDER])
    ax.tick_params(axis="x", labelsize=int(11 * FONT_SCALE))
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.subplots_adjust(right=0.88)
    ax.legend(handles=model_legend_handles(), title="Model", frameon=False,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    safe_savefig(plt.gcf(), outpath)
    plt.close()


def plot_auroc_grouped(df_task, title, outpath):
    _plot_grouped(df_task, title, outpath,
                  y_col="auroc", ylim=AUROC_YLIM, ylabel="AUROC", baseline=0.5)


def plot_spearman_grouped(df_task, title, outpath):
    _plot_grouped(df_task, title, outpath,
                  y_col="spearman_rho_boot_mean", ylim=SPEARMAN_YLIM,
                  ylabel="Spearman ρ\n(bootstrap mean)", baseline=0.0)


# ---------------------------------------------------------------------
# Grouped plots
# ---------------------------------------------------------------------
for task in df_main["task"].unique():
    plot_auroc_grouped(df_main[df_main["task"] == task],
                       f"Phase 2 AUROC + 95% CI (grouped) — {task}",
                       FIGS / f"fig_phase2_auroc_grouped_{task}.pdf")
plot_auroc_grouped(df_main,
                   "Phase 2 AUROC + 95% CI (grouped) — all runs\n(MedQA + PubMedQA shown together)",
                   FIGS / "fig_phase2_auroc_grouped_ALL.pdf")

for task in df_spear_filtered["task"].unique():
    plot_spearman_grouped(df_spear_filtered[df_spear_filtered["task"] == task],
                          f"Phase 2 Spearman ρ + 95% CI (grouped) — {task}",
                          FIGS / f"fig_phase2_spearman_grouped_{task}.pdf")
plot_spearman_grouped(df_spear_filtered,
                      "Phase 2 Spearman ρ + 95% CI (grouped) — all runs\n(MedQA + PubMedQA shown together)",
                      FIGS / "fig_phase2_spearman_grouped_ALL.pdf")

# ======================================================================
# STORY FIGURES (1–4)
# ======================================================================
TASK_ORDER_STORY  = ["medqa", "pubmedqa"]
MODEL_ORDER_STORY = ["mistral", "biomistral"]
TASK_PRETTY       = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Ternary)"}
STORY_DIR         = FIGS / "story"
STORY_DIR.mkdir(parents=True, exist_ok=True)


def _panel_bar(ax, sub, title):
    sub = sub.set_index("score").reindex(SCORE_ORDER).reset_index()
    x   = np.arange(len(SCORE_ORDER), dtype=float)
    y   = sub["auroc"].to_numpy(dtype=float)
    lo  = sub["ci95_lo"].to_numpy(dtype=float)
    hi  = sub["ci95_hi"].to_numpy(dtype=float)
    ax.bar(x, y, width=0.65)
    _errorbar(ax, x, y, y - lo, hi - y)
    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in SCORE_ORDER], rotation=0, fontsize=14)
    ax.set_ylim(*AUROC_YLIM)
    ax.set_title(title)
    ax.set_ylabel("AUROC")
    add_value_labels_above_ci(ax, x, y, hi - y)


def _panel_delta(ax, sub, title, y0=-0.05, y1=0.30):
    sub = sub.set_index("score").reindex(SCORE_ORDER).reset_index()
    x   = np.arange(len(SCORE_ORDER), dtype=float)
    y   = sub["auroc"].to_numpy(dtype=float) - 0.5
    lo  = sub["ci95_lo"].to_numpy(dtype=float) - 0.5
    hi  = sub["ci95_hi"].to_numpy(dtype=float) - 0.5
    ax.bar(x, y, width=0.65)
    _errorbar(ax, x, y, y - lo, hi - y)
    ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in SCORE_ORDER], rotation=0, fontsize=14)
    ax.set_ylim(y0, y1)
    ax.set_title(title)
    ax.set_ylabel("ΔAUROC (vs 0.5)", fontsize=ax.xaxis.get_label().get_size())
    add_value_labels_above_ci(ax, x, y, hi - y)


def _story_2x2(panel_fn, suptitle, outfile):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88)
    for i, task in enumerate(TASK_ORDER_STORY):
        for j, model in enumerate(MODEL_ORDER_STORY):
            sub = df_main[(df_main["task"] == task) & (df_main["model"] == model)].copy()
            panel_fn(axes[i, j], sub, f"{TASK_PRETTY[task]} — {MODEL_PRETTY[model]}")
    fig.suptitle(suptitle, y=1.01, fontsize=plt.rcParams["figure.titlesize"])
    safe_savefig(fig, outfile, bbox_inches="tight")
    plt.close(fig)


_story_2x2(_panel_bar,
           "Phase 2: White-box scorers across Task × Model (AUROC ± 95% CI)",
           STORY_DIR / "fig_phase2_story_1_grid_task_model_auroc.pdf")

_story_2x2(_panel_delta,
           "Phase 2: Effect size vs random (ΔAUROC ± 95% CI)",
           STORY_DIR / "fig_phase2_story_2_grid_task_model_delta_auroc.pdf")

# Story 3: task-format effect lines
key_scores = ["lntp", "egh_probe_oof", "hidden_probe_oof"]
fig, axes  = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
for j, model in enumerate(MODEL_ORDER_STORY):
    ax = axes[j]
    for score in key_scores:
        ys, los, his = [], [], []
        for task in TASK_ORDER_STORY:
            r = df_main[(df_main["task"] == task) & (df_main["model"] == model) &
                        (df_main["score"] == score)]
            if r.empty: raise KeyError(f"Missing row: task={task}, model={model}, score={score}")
            r = r.sort_values("auroc", ascending=False).iloc[0]
            ys.append(float(r["auroc"])); los.append(float(r["ci95_lo"])); his.append(float(r["ci95_hi"]))
        x = np.arange(len(TASK_ORDER_STORY), dtype=float)
        ax.plot(x, ys, marker="o", label=pretty_score(score))
        _errorbar(ax, x, ys, np.array(ys) - np.array(los), np.array(his) - np.array(ys))
    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(np.arange(len(TASK_ORDER_STORY)))
    ax.set_xticklabels([TASK_PRETTY[t] for t in TASK_ORDER_STORY],
                       rotation=0, fontsize=int(11.5 * FONT_SCALE))
    ax.set_title(MODEL_PRETTY[model])
    ax.set_ylabel("AUROC", fontsize=int(11.5 * FONT_SCALE))
    ax.set_ylim(*AUROC_YLIM)
axes[0].legend(frameon=False, title="Scorer")
fig.suptitle("Task-format effect: scorer performance shifts from MCQ → Yes/No",
             y=1.12, fontsize=plt.rcParams["figure.titlesize"])
safe_savefig(fig, STORY_DIR / "fig_phase2_story_3_task_format_effect_lines.pdf", bbox_inches="tight")
plt.close(fig)

# Story 4: model specialisation effect
rows = []
for task in TASK_ORDER_STORY:
    for score in key_scores:
        r_m = df_main[(df_main["task"] == task) & (df_main["model"] == "mistral")    & (df_main["score"] == score)]
        r_b = df_main[(df_main["task"] == task) & (df_main["model"] == "biomistral") & (df_main["score"] == score)]
        if r_m.empty or r_b.empty:
            raise KeyError(f"Missing rows for diff: task={task}, score={score}")
        r_m, r_b = r_m.iloc[0], r_b.iloc[0]
        diff = float(r_m["auroc"]) - float(r_b["auroc"])
        rows.append({"task": task, "score": score, "diff": diff,
                     "lo": float(r_m["ci95_lo"]) - float(r_b["ci95_hi"]),
                     "hi": float(r_m["ci95_hi"]) - float(r_b["ci95_lo"])})
dd = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
x_labels, vals, err_low, err_high = [], [], [], []
for task in TASK_ORDER_STORY:
    for score in key_scores:
        r = dd[(dd["task"] == task) & (dd["score"] == score)].iloc[0]
        x_labels.append(f"{TASK_PRETTY[task]}\n{pretty_score(score)}")
        vals.append(float(r["diff"]))
        err_low.append(float(r["diff"] - r["lo"]))
        err_high.append(float(r["hi"] - r["diff"]))

x = np.arange(len(vals), dtype=float)
ax.bar(x, vals, width=0.65)
_errorbar(ax, x, vals, err_low, err_high)
ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=20, ha="right")
ax.tick_params(axis="x", pad=6)
ax.set_ylabel("ΔAUROC (Mistral − BioMistral)")
ax.set_title("Model specialization effect (approx. CI via bounds)", pad=18)
ymin = min(np.array(vals) - np.array(err_low))
ymax = max(np.array(vals) + np.array(err_high))
pad  = 0.12 * (ymax - ymin + 1e-9)
ax.set_ylim(ymin - 0.15 * pad, ymax + pad)
add_value_labels_above_ci(ax, x, vals, err_high,
                           fontsize=int(12 * FONT_SCALE), pad_frac=0.025)
safe_savefig(fig, STORY_DIR / "fig_phase2_story_4_model_diff_delta_auroc.pdf", bbox_inches="tight")
plt.close(fig)