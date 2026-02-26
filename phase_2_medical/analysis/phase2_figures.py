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
from sklearn.metrics import roc_auc_score

import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


FONT_SCALE = 1.5  # Global typography knob; keep fixed for cross-figure comparability in exported PDFs.

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],

    "font.size": int(12 * FONT_SCALE),
    "axes.titlesize": int(13 * FONT_SCALE),
    "axes.labelsize": int(12 * FONT_SCALE),
    "xtick.labelsize": int(12 * FONT_SCALE),
    "ytick.labelsize": int(11 * FONT_SCALE),
    "legend.fontsize": int(11 * FONT_SCALE),
    "legend.title_fontsize": int(11 * FONT_SCALE),
    "figure.titlesize": int(14.5 * FONT_SCALE),

    "axes.titlepad": 12,
    
    # Slightly thicker axes/ticks for print/PDF legibility.
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,

    # Vector-friendly fonts for downstream editing/review (avoid Type 3 fonts).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE) # Value labels above CI whiskers; intentionally smaller than tick labels.


# ---------------------------------------------------------------------
# Paths (FINAL-only; Ablations gehören NICHT hier rein)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]   # Repo root for phase_2_medical; used to resolve outputs deterministically.
FINAL = ROOT / "outputs" / "final"
FIGS = ROOT / "outputs" / "figures_tables" / "figures_general"
FIGS.mkdir(parents=True, exist_ok=True)

# Canonical Phase-2 scorer keys; only these are aggregated/plotted (stabilizes reported results).
MAIN_SCORES = {"lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"}

# ---------------------------------------------------------------------
# Plot constants (for consistent scaling across ALL figures)
# ---------------------------------------------------------------------
AUROC_YLIM = (0.45, 0.80)        # Fixed y-range to prevent scale-driven comparability issues across tasks/models.
SPEARMAN_YLIM = (-0.05, 0.60)    # Fixed y-range for correlation figures (comparability > tight autoscaling).

# ---------------------------------------------------------------------
# Plot styling knobs (global, for print/readability)
# ---------------------------------------------------------------------
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

# Canonical display labels for scorer keys (stabilizes figure text across pipelines).
SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}
SCORE_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]  # Stable reviewer-facing order in grouped panels.

MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}

MODEL_COLOR = {
    "mistral": "tab:blue",
    "biomistral": "tab:orange",
}

# ---------------------------------------------------------------------
# Robust save helper (handles Windows PDF file locks)
# ---------------------------------------------------------------------
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a figure; if the target is locked, write to *_v{k}.pdf instead."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        # NOTE: potential issue: open-file locks can change filenames via _v{k}, complicating scripted collection.
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_jsonl(path: Path):
    """Load JSONL into a list[dict]; ignores blank lines."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_bootstrap_map(npz_path: Path):
    """
   Load bootstrap index arrays from NPZ into a dict[str, np.ndarray].

    Accepts both multi-key NPZs (per-scorer arrays) and legacy single-array files; in the
    latter case the array is also exposed under 'indices' for downstream compatibility.
    """
    z = np.load(npz_path, allow_pickle=True)
    out = {}

    for k in z.files:
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack(arr, axis=0)
        # Keep only array-like contents; non-arrays are ignored to avoid schema-dependent branching.
        if isinstance(arr, np.ndarray):
            out[str(k).lower()] = arr.astype(int)

    # Backward compatibility: if there is a common single key, alias to 'indices'.
    for alias in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if alias in out and "indices" not in out:
            out["indices"] = out[alias]

    # If it's truly just a single array with some unknown key, also provide 'indices'.
    if "indices" not in out and len(out) == 1:
        out["indices"] = next(iter(out.values()))

    return out

def find_label_key(example: dict):
    """Infer the binary label field name using fixed precedence (guards against schema drift)."""
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    """Extract per-scorer numeric values from nested dicts or via heuristic top-level key matching."""
    # Preferred schemas: scores/wb_scores dictionaries emitted by the evaluation runner.
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            # Heuristic: accept numeric fields whose key contains a known scorer identifier.
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)
    return scores

def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and choose a score polarity that yields AUROC >= 0.5 when possible."""
    au = roc_auc_score(y, s)
    if au < 0.5:
        # Polarity convention: flip sign so "higher is better" for plotting and for downstream Spearman.
        return roc_auc_score(y, -s), -1.0
    return au, +1.0

def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """
   Percentile bootstrap CI for AUROC using precomputed resample indices.

    Resamples are defined by boot_idx (no RNG). Degenerate resamples (single-class y) are
    skipped; if all resamples are degenerate the returned mean/CI are NaN.
    """
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: AUROC undefined if only one class is present.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))

    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        # NOTE: potential issue: all resamples degenerate → CI/mean become NaN and propagate to plots.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi

def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """
   Percentile bootstrap CI for Spearman ρ using precomputed resample indices.

    Skips resamples with constant y (undefined correlation) and any NaN correlations; if all
    resamples are invalid the returned mean/CI are NaN.
    """
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: correlation undefined if one vector is constant.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        if pd.isna(rho):
            continue
        rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0:
        # NOTE: potential issue: all resamples invalid → CI/mean become NaN and propagate to plots.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi

def infer_task_model_from_manifest(manifest_path: Path):
    """Infer (task, model) identifiers from a Phase-2 manifest.json schema."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(m.get("task", "")).lower()
    config = m.get("config", {})
    model_name = str(config.get("model_name", "")).lower()
    # Heuristic grouping: any model_name containing "bio" is bucketed as "biomistral" for plots.
    model = "biomistral" if "bio" in model_name else "mistral"
    return task, model

def pretty_score(score_key: str) -> str:
    """Map scorer keys to stable publication-facing labels."""
    k = str(score_key).lower()
    return SCORE_PRETTY.get(k, k)


def model_legend_handles():
    return [
        Line2D(
            [0], [0],
            color=MODEL_COLOR["mistral"],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=MODEL_PRETTY.get("mistral", "Mistral"),
        ),
        Line2D(
            [0], [0],
            color=MODEL_COLOR["biomistral"],
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=6.0,
            label=MODEL_PRETTY.get("biomistral", "BioMistral"),
        ),
    ]


def add_model_legend_between_title_and_plot(fig, ncol=2, y=0.925):
    fig.legend(
        handles=model_legend_handles(),
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
 
# ---------------------------------------------------------------------
# Value labels (3 decimals) above CI whiskers
# ---------------------------------------------------------------------
def add_value_labels_above_ci(ax, x_positions, y_values, yerr_high, fmt="{:.3f}", fontsize=None, pad_frac=0.0135):
    """Annotate values above the upper CI whisker (y + yerr_high) with axis-relative padding."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span  # Pad in axis units so labels remain consistent under fixed y-limits.

    for x, y, eh in zip(x_positions, y_values, yerr_high):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        top = y + (0.0 if eh is None else float(eh))
        ax.text(float(x), float(top) + pad, fmt.format(float(y)),
                ha="center", va="bottom", fontsize=fontsize)

# ---------------------------------------------------------------------
# Collect FINAL runs
# ---------------------------------------------------------------------
print("ROOT =", ROOT)
print("FINAL =", FINAL)
print("FINAL exists =", FINAL.exists())

if not FINAL.exists():
    raise FileNotFoundError(f"FINAL directory not found: {FINAL}")

runs = []
for manifest_path in sorted(FINAL.glob("*.manifest.json")):
    # Naming invariant: artifacts share the manifest prefix (only the suffix differs).
    results_path = manifest_path.with_suffix("").with_suffix(".results.jsonl")
    boot_path = manifest_path.with_suffix("").with_suffix(".manifest.bootstrap_indices.npz")

    if not results_path.exists():
        # Summary warning only: missing runs are excluded from aggregation/plots.
        print("[WARN] Missing results for", manifest_path.name, "expected:", results_path.name)
        continue
    if not boot_path.exists():
        # Summary warning only: missing runs are excluded from aggregation/plots.
        print("[WARN] Missing bootstrap npz for", manifest_path.name, "expected:", boot_path.name)
        continue

    task, model = infer_task_model_from_manifest(manifest_path)
    runs.append((task, model, results_path, manifest_path, boot_path))

print("Found FINAL runs:", [(t, m, p.name) for t, m, p, _, _ in runs])

if len(runs) == 0:
    listing = sorted([p.name for p in FINAL.iterdir()])
    raise RuntimeError(
        "Found runs: [] but FINAL contains files. This usually means naming mismatch.\n"
        f"FINAL listing:\n{listing}\n"
        "Expected pattern per run:\n"
        "  *.manifest.json\n"
        "  sameprefix.results.jsonl\n"
        "  sameprefix.manifest.bootstrap_indices.npz\n"
    )

# ---------------------------------------------------------------------
# Compute AUROC + CI per score
# ---------------------------------------------------------------------
records = []
spearman_records = []
spearman_ci_rows = []

for task, model, results_path, manifest_path, boot_path in runs:
    rows = load_jsonl(results_path)
    if not rows:
        # Empty results contribute nothing to aggregates; no placeholder rows are written.
        continue

    # Label key is inferred once from rows[0]; assumes schema consistency within the JSONL file.
    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # Score extraction: normalize keys to lowercase to avoid casing mismatches across pipelines.
    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]

    # Alignment invariant: only keep score keys present in *all* rows (prevents implicit NaNs/misalignment).
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted(keys)

    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
    # Report only canonical Phase-2 scorers; other numeric fields are intentionally dropped.
    S = {k: v for k, v in S.items() if k in MAIN_SCORES}
   
    # Reproducibility: bootstrap indices are loaded verbatim (no RNG in this script).
    boot_map = load_bootstrap_map(boot_path)
    hidden_kept = boot_map.get("hidden_kept_indices", None)

    # Mapping from scorer key to NPZ array key; supports older/newer naming conventions.
    SCORE_TO_BOOTKEY = {
        "lntp": "lntp",
        "mtp": "mtp",

        # EGH family
        "egh_probe_oof": "egh",
        "egh_probe_ge": "egh_ge",
        "egh_probe_g_only": "egh_g",
        "egh_probe_e_only": "egh_e",
        "egh_probe_scalar_only": "egh_scalar",

        # Hidden
        "hidden_probe_oof": "hidden",
    }
    
    for score_name, s_raw in S.items():
        # Polarity convention: choose direction so AUROC >= 0.5 when possible; reuse for derived stats/plots.
        score_l = str(score_name).lower()

        # --- choose bootstrap indices for this score ---
        boot_key = SCORE_TO_BOOTKEY.get(score_l, score_l)
        boot_idx = boot_map.get(boot_key, None)

        # Backward compatibility: if per-score key missing, try global 'indices'.
        if boot_idx is None:
            boot_idx = boot_map.get("indices", None)

        if boot_idx is None:
            print(f"[WARN] No bootstrap indices for score={score_name} in {boot_path.name}; skipping.")
            continue

        # --- choose direction (hidden must be oriented on the kept subset) ---
        if score_l == "hidden_probe_oof" and hidden_kept is not None:
            # Invariant: direction must be fit on the same subset that will be bootstrapped/plotted.
            au, direction = auroc_with_best_direction(y[hidden_kept], s_raw[hidden_kept])
        else:
            au, direction = auroc_with_best_direction(y, s_raw)

        s = s_raw * direction  # Apply chosen polarity once; downstream metrics assume "higher is better".

        # --- choose population consistent with bootstrap indices ---
        if score_l == "hidden_probe_oof":
            if hidden_kept is None:
                # Fallback: drop non-finite scores to avoid NaNs in AUROC/correlation and index mismatch.
                m = np.isfinite(s)
                y_use = y[m]
                s_use = s[m]
            else:
                y_use = y[hidden_kept]
                s_use = s[hidden_kept]
        else:
            y_use = y
            s_use = s
                    
        
        if boot_idx.shape[1] != len(y_use):
            # Fail fast: otherwise resampling silently indexes the wrong population and invalidates CIs.
            raise ValueError(
                f"Bootstrap shape mismatch for {score_l}: boot_idx {boot_idx.shape} vs N={len(y_use)} "
                f"(boot_file={boot_path.name}, boot_key={boot_key})"
            )

        # --- AUROC CI on consistent population ---
        mean_b, lo, hi = bootstrap_ci_from_indices(y_use, s_use, boot_idx, alpha=0.05)

        # --- Spearman (point + CI) on consistent population ---
        # NOTE: potential issue: Spearman uses y as numeric-coded labels; interpret as rank association, not calibration.
        rho = pd.Series(s_use).corr(pd.Series(y_use), method="spearman")
        
        spearman_records.append({
            "task": task,
            "model": model,
            "score": score_l,
            "spearman_rho": float(rho),
            "direction": float(direction),
            "N": int(len(y_use)),
            "pos_rate": float(np.mean(y_use)),
        })

        m_rho, lo_rho, hi_rho = bootstrap_spearman_ci_from_indices(y_use, s_use, boot_idx, alpha=0.05)
        spearman_ci_rows.append({
            "task": task,
            "model": model,
            "score": score_l,
            "spearman_rho_boot_mean": m_rho,
            "ci95_lo": lo_rho,
            "ci95_hi": hi_rho,
        })
        
        # --- store AUROC rows (for AUROC plots) ---
        records.append({
            "task": task,
            "model": model,
            "score": score_l,

            "direction": float(direction),
            "N": int(len(y_use)),
            "pos_rate": float(np.mean(y_use)),

            "auroc": float(au),
            "auroc_boot_mean": float(mean_b),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),

            "manifest_file": str(manifest_path),
            "results_file": str(results_path),
            "boot_file": str(boot_path),
        })

df = pd.DataFrame(records).sort_values(["task", "model", "auroc"], ascending=[True, True, False])

# Keep dataframes in-memory for plotting (no CSV output).
df_main = df[df["score"].isin(MAIN_SCORES)].copy()

df_spear_main = pd.DataFrame(spearman_records)
df_spear_ci = pd.DataFrame(spearman_ci_rows)
# Left-merge preserves point estimates even if CI computation produced NaNs for a run/score.
df_spear_main = df_spear_main.merge(df_spear_ci, on=["task", "model", "score"], how="left")

# Plot: AUROC bar + CI (labels above CI; errorbars black)
# ---------------------------------------------------------------------
def plot_auroc_bar(df_task: pd.DataFrame, title: str, outpath: Path):
    dfp = df_task.copy()
    dfp["task"] = dfp["task"].astype(str).str.lower()
    dfp["model"] = dfp["model"].astype(str).str.lower()
    dfp["score"] = dfp["score"].astype(str).str.lower()

    multi_task = dfp["task"].nunique() > 1
    if multi_task:
        dfp["label"] = dfp["task"].str.upper() + " | " + dfp["score"].map(pretty_score)
    else:
        dfp["label"] = dfp["score"].map(pretty_score)

    dfp = dfp.sort_values("auroc_boot_mean", ascending=False).reset_index(drop=True)

    x = np.arange(len(dfp), dtype=float)
    yv = dfp["auroc_boot_mean"].to_numpy(dtype=float)

    lo = dfp["ci95_lo"].to_numpy(dtype=float)
    hi = dfp["ci95_hi"].to_numpy(dtype=float)
    yerr_low = yv - lo
    yerr_high = hi - yv

    n_bars = len(dfp)
    fig_w = max(9.5, n_bars * 0.78)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))


    bar_colors = [MODEL_COLOR.get(m, "tab:gray") for m in dfp["model"]]
    ax.bar(x, yv, width=0.65, color=bar_colors)
    ax.errorbar(
        x, yv, yerr=[yerr_low, yerr_high],
        fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
    )

    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_ylim(*AUROC_YLIM)
    ax.set_ylabel("AUROC\n(bootstrap mean)")
    ax.yaxis.label.set_size(int(10.5 * FONT_SCALE)) 
    fig.suptitle(title, y=0.975)
    add_model_legend_between_title_and_plot(fig, ncol=2, y=0.915)

    ax.set_xticks(x)
    if multi_task:
        ax.set_xticklabels(dfp["label"].tolist(), rotation=35, ha="right")
    else:
        ax.set_xticklabels(dfp["label"].tolist(), rotation=0, ha="center")
        
    ax.tick_params(axis="x", labelsize=int(10 * FONT_SCALE))

    # Value labels
    add_value_labels_above_ci(ax, x, yv, yerr_high, fmt="{:.3f}")

    # ONE layout + ONE save
    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    

# ---------------------------------------------------------------------
# AUROC bar plots (single-task + ALL)
# ---------------------------------------------------------------------
df_main_filtered = df_main[df_main["score"].isin(MAIN_SCORES)].copy()

for task in df_main_filtered["task"].unique():
    plot_auroc_bar(
        df_main_filtered[df_main_filtered["task"] == task],
        f"Phase 2 AUROC + 95% CI — {task}",
        FIGS / f"fig_phase2_auroc_bar_{task}.pdf"
    )

plot_auroc_bar(
    df_main_filtered,
    "Phase 2 AUROC + 95% CI — all runs (MedQA + PubMedQA shown together)",
    FIGS / "fig_phase2_auroc_bar_ALL.pdf"
)

# ---------------------------------------------------------------------
# Plot: Spearman bar + CI (labels above CI; errorbars black)
# ---------------------------------------------------------------------
def plot_spearman_bar(df_task: pd.DataFrame, title: str, outpath: Path):
    dfp = df_task.copy()
    dfp["task"] = dfp["task"].astype(str).str.lower()
    dfp["model"] = dfp["model"].astype(str).str.lower()
    dfp["score"] = dfp["score"].astype(str).str.lower()

    multi_task = dfp["task"].nunique() > 1
    if multi_task:
        dfp["label"] = dfp["task"].str.upper() + " | " + dfp["score"].map(pretty_score)
    else:
        dfp["label"] = dfp["score"].map(pretty_score)

    dfp = dfp.sort_values("spearman_rho_boot_mean", ascending=False).reset_index(drop=True)

    x = np.arange(len(dfp), dtype=float)
    yv = dfp["spearman_rho_boot_mean"].to_numpy(dtype=float)

    lo = dfp["ci95_lo"].to_numpy(dtype=float)
    hi = dfp["ci95_hi"].to_numpy(dtype=float)
    yerr_low = yv - lo
    yerr_high = hi - yv

    n_bars = len(dfp)
    fig_w = max(9.5, n_bars * 0.78)
    fig, ax = plt.subplots(figsize=(fig_w, 7.5))


    bar_colors = [MODEL_COLOR.get(m, "tab:gray") for m in dfp["model"]]
    ax.bar(x, yv, width=0.65, color=bar_colors)
    ax.errorbar(
        x, yv, yerr=[yerr_low, yerr_high],
        fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
    )

    ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_ylim(*SPEARMAN_YLIM)
    ax.set_ylabel("Spearman ρ\n(bootstrap mean)")
    ax.yaxis.label.set_size(int(10.5 * FONT_SCALE)) 

    fig.suptitle(title, y=0.975)
    add_model_legend_between_title_and_plot(fig, ncol=2, y=0.915)
    
    ax.set_xticks(x)
    if multi_task:
        ax.set_xticklabels(dfp["label"].tolist(), rotation=35, ha="right")
    else:
        ax.set_xticklabels(dfp["label"].tolist(), rotation=0, ha="center")
    
    ax.tick_params(axis="x", labelsize=int(10 * FONT_SCALE))
        
    add_value_labels_above_ci(ax, x, yv, yerr_high, fmt="{:.3f}")

    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
        

# ---------------------------------------------------------------------
# Spearman bar plots (single-task + ALL)
# ---------------------------------------------------------------------
df_spear_main_filtered = df_spear_main[df_spear_main["score"].isin(MAIN_SCORES)].copy()
for task in df_spear_main_filtered["task"].unique():
    plot_spearman_bar(
        df_spear_main_filtered[df_spear_main_filtered["task"] == task],
        f"Phase 2 Spearman ρ + 95% CI — {task}",
        FIGS / f"fig_phase2_spearman_bar_{task}.pdf"
    )
plot_spearman_bar(
    df_spear_main_filtered,
    "Phase 2 Spearman ρ + 95% CI — all runs (MedQA + PubMedQA shown together)",
    FIGS / "fig_phase2_spearman_bar_ALL.pdf"
)

# ======================================================================
# Grouped plots (labels above CI; errorbars black)
# ======================================================================
MODEL_ORDER = ["mistral", "biomistral"]  # Stable ordering for grouped comparisons (avoids implicit sorting drift).

def plot_auroc_grouped(df_task: pd.DataFrame, title: str, outpath: Path):
    """Plot grouped AUROC bars by scorer with per-model bars and 95% CIs."""
    dfp = df_task.copy()
    dfp["score"] = dfp["score"].map(lambda x: str(x).lower())
    dfp["model"] = dfp["model"].map(lambda x: str(x).lower())
    dfp = dfp[dfp["score"].isin(SCORE_ORDER)].copy()

    scorers = SCORE_ORDER
    models = [m for m in MODEL_ORDER if m in set(dfp["model"])]

    def get_row(score, model):
        sub = dfp[(dfp["score"] == score) & (dfp["model"] == model)]
        if len(sub) == 0:
            return None
        # Convention: if multiple runs exist for a (score, model), display the best-performing run.
        return sub.sort_values("auroc", ascending=False).iloc[0]

    x_base = np.arange(len(scorers), dtype=float)
    width = 0.38 if len(models) == 2 else 0.6  # Visual heuristic: keep groups compact for two-model comparisons.

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        xs = x_base + offset

        ys, yerr_low, yerr_high = [], [], []
        for s in scorers:
            r = get_row(s, model)
            if r is None:
                # Missing runs are rendered as NaN with zero error (explicitly signals absent comparisons).
                ys.append(np.nan); yerr_low.append(0.0); yerr_high.append(0.0)
            else:
                ys.append(float(r["auroc"]))
                yerr_low.append(float(r["auroc"]) - float(r["ci95_lo"]))
                yerr_high.append(float(r["ci95_hi"]) - float(r["auroc"]))

        ax.bar(xs, ys, width=width * 0.95, color=MODEL_COLOR.get(model, "tab:gray"))
        ax.errorbar(
            xs, ys, yerr=[yerr_low, yerr_high],
            fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
            elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
        )
        add_value_labels_above_ci(ax, xs, ys, yerr_high, fmt="{:.3f}", pad_frac=0.010)

    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x_base)
    ax.set_xticklabels([pretty_score(s) for s in scorers])
    ax.tick_params(axis="x", labelsize=int(11 * FONT_SCALE))
    ax.set_ylim(*AUROC_YLIM)
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    plt.subplots_adjust(right=0.88)
    ax.legend(
        handles=model_legend_handles(),
        title="Model",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0)
    )

    plt.tight_layout()
    safe_savefig(plt.gcf(), outpath)
    plt.close()

def plot_spearman_grouped(df_task: pd.DataFrame, title: str, outpath: Path):
    """Plot grouped bootstrap-mean Spearman ρ bars by scorer with per-model bars and 95% CIs."""
    dfp = df_task.copy()
    dfp["score"] = dfp["score"].map(lambda x: str(x).lower())
    dfp["model"] = dfp["model"].map(lambda x: str(x).lower())
    dfp = dfp[dfp["score"].isin(SCORE_ORDER)].copy()

    scorers = SCORE_ORDER
    models = [m for m in MODEL_ORDER if m in set(dfp["model"])]

    def get_row(score, model):
        sub = dfp[(dfp["score"] == score) & (dfp["model"] == model)]
        if len(sub) == 0:
            return None
        # Convention: if multiple runs exist for a (score, model), display the best bootstrap-mean ρ.
        return sub.sort_values("spearman_rho_boot_mean", ascending=False).iloc[0]

    x_base = np.arange(len(scorers), dtype=float)
    width = 0.38 if len(models) == 2 else 0.6  # Visual heuristic: keep groups compact for two-model comparisons.

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        xs = x_base + offset

        ys, yerr_low, yerr_high = [], [], []
        for s in scorers:
            r = get_row(s, model)
            if r is None:
                # Missing runs are rendered as NaN with zero error (explicitly signals absent comparisons).
                ys.append(np.nan); yerr_low.append(0.0); yerr_high.append(0.0)
            else:
                ys.append(float(r["spearman_rho_boot_mean"]))
                yerr_low.append(float(r["spearman_rho_boot_mean"]) - float(r["ci95_lo"]))
                yerr_high.append(float(r["ci95_hi"]) - float(r["spearman_rho_boot_mean"]))

        ax.bar(xs, ys, width=width * 0.95, color=MODEL_COLOR.get(model, "tab:gray"))
        ax.errorbar(
            xs, ys, yerr=[yerr_low, yerr_high],
            fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
            elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
        )
        add_value_labels_above_ci(ax, xs, ys, yerr_high, fmt="{:.3f}", pad_frac=0.010)

    ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x_base)
    ax.set_xticklabels([pretty_score(s) for s in scorers])
    ax.tick_params(axis="x", labelsize=int(11 * FONT_SCALE))
    ax.set_ylim(*SPEARMAN_YLIM)
    ax.set_ylabel("Spearman ρ\n(bootstrap mean)")
    ax.set_title(title)
    plt.subplots_adjust(right=0.88)
    ax.legend(
        handles=model_legend_handles(),
        title="Model",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0)
    )

    plt.tight_layout()
    safe_savefig(plt.gcf(), outpath)
    plt.close()

for task in df_main["task"].unique():
    plot_auroc_grouped(df_main[df_main["task"] == task],
                       f"Phase 2 AUROC + 95% CI (grouped) — {task}",
                       FIGS / f"fig_phase2_auroc_grouped_{task}.pdf")
plot_auroc_grouped(df_main,
                   "Phase 2 AUROC + 95% CI (grouped) — all runs\n(MedQA + PubMedQA shown together)",
                   FIGS / "fig_phase2_auroc_grouped_ALL.pdf")

for task in df_spear_main_filtered["task"].unique():
    plot_spearman_grouped(df_spear_main_filtered[df_spear_main_filtered["task"] == task],
                          f"Phase 2 Spearman ρ + 95% CI (grouped) — {task}",
                          FIGS / f"fig_phase2_spearman_grouped_{task}.pdf")
plot_spearman_grouped(df_spear_main_filtered,
                      "Phase 2 Spearman ρ + 95% CI (grouped) — all runs\n(MedQA + PubMedQA shown together)",
                      FIGS / "fig_phase2_spearman_grouped_ALL.pdf")

# ======================================================================
# STORY FIGURES (1–4)
# - Only change: add value labels above CI whiskers + black errorbars where applicable
# ======================================================================
TASK_ORDER_STORY = ["medqa", "pubmedqa"]
MODEL_ORDER_STORY = ["mistral", "biomistral"]

TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}


def _panel_bar(ax, sub, title):
    """Single panel: AUROC bars with 95% CIs for a fixed (task, model)."""
    # Invariant: reindex to SCORE_ORDER so panels remain comparable even if df ordering changes.
    sub = sub.set_index("score").reindex(SCORE_ORDER).reset_index()
    x = np.arange(len(SCORE_ORDER), dtype=float)
    y = sub["auroc"].to_numpy(dtype=float)
    lo = sub["ci95_lo"].to_numpy(dtype=float)
    hi = sub["ci95_hi"].to_numpy(dtype=float)
    yerr_low = y - lo
    yerr_high = hi - y
    yerr = np.vstack([yerr_low, yerr_high])

    ax.bar(x, y, width=0.65)
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
    )
    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in SCORE_ORDER], rotation=0, fontsize=14)
    ax.set_ylim(*AUROC_YLIM)
    ax.set_title(title)
    ax.set_ylabel("AUROC")

    add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

def _panel_delta(ax, sub, title, y0=-0.05, y1=0.30):
    """Single panel: ΔAUROC vs the 0.5 baseline (effect size) with 95% CIs."""
    # Invariant: reindex to SCORE_ORDER so panels remain comparable even if df ordering changes.
    sub = sub.set_index("score").reindex(SCORE_ORDER).reset_index()
    x = np.arange(len(SCORE_ORDER), dtype=float)

    # Effect size expressed as AUROC - 0.5 (chance baseline).
    y = sub["auroc"].to_numpy(dtype=float) - 0.5
    lo = sub["ci95_lo"].to_numpy(dtype=float) - 0.5
    hi = sub["ci95_hi"].to_numpy(dtype=float) - 0.5
    yerr_low = y - lo
    yerr_high = hi - y
    yerr = np.vstack([yerr_low, yerr_high])

    ax.bar(x, y, width=0.65)
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
        elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
    )
    ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_score(s) for s in SCORE_ORDER], rotation=0, fontsize=14)
    ax.set_ylim(y0, y1)
    ax.set_title(title)
    ax.set_ylabel("ΔAUROC (vs 0.5)", fontsize=ax.xaxis.get_label().get_size())

    add_value_labels_above_ci(ax, x, y, yerr_high, fmt="{:.3f}")

STORY_DIR = FIGS / "story"
STORY_DIR.mkdir(parents=True, exist_ok=True)

# (1) 2×2 Grid AUROC
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88)
for i, task in enumerate(TASK_ORDER_STORY):
    for j, model in enumerate(MODEL_ORDER_STORY):
        ax = axes[i, j]
        sub = df_main[(df_main["task"] == task) & (df_main["model"] == model)].copy()
        _panel_bar(ax, sub, f"{TASK_PRETTY[task]} — {MODEL_PRETTY[model]}")
fig.suptitle("Phase 2: White-box scorers across Task × Model (AUROC ± 95% CI)", y=1.01, fontsize=plt.rcParams["figure.titlesize"])
safe_savefig(fig, STORY_DIR / "fig_phase2_story_1_grid_task_model_auroc.pdf", bbox_inches="tight")
plt.close(fig)

# (2) 2×2 Grid ΔAUROC
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.88)
for i, task in enumerate(TASK_ORDER_STORY):
    for j, model in enumerate(MODEL_ORDER_STORY):
        ax = axes[i, j]
        sub = df_main[(df_main["task"] == task) & (df_main["model"] == model)].copy()
        _panel_delta(ax, sub, f"{TASK_PRETTY[task]} — {MODEL_PRETTY[model]}")
fig.suptitle("Phase 2: Effect size vs random (ΔAUROC ± 95% CI)", y=1.01, fontsize=plt.rcParams["figure.titlesize"])
safe_savefig(fig, STORY_DIR / "fig_phase2_story_2_grid_task_model_delta_auroc.pdf", bbox_inches="tight")
plt.close(fig)

# (3) Task-format effect lines (unchanged plot type; only make errorbars black)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
key_scores = ["lntp", "egh_probe_oof", "hidden_probe_oof"]  # Subset for narrative focus; not an exhaustive scorer list.

for j, model in enumerate(MODEL_ORDER_STORY):
    ax = axes[j]
    for score in key_scores:
        ys, los, his = [], [], []
        for task in TASK_ORDER_STORY:
            sub_row = df_main[
                (df_main["task"] == task) &
                (df_main["model"] == model) &
                (df_main["score"] == score)
            ]
            if sub_row.empty:
                # Paired lines imply direct across-task comparisons; missing rows would mislead, so fail fast.
                raise KeyError(f"Missing row for task={task}, model={model}, score={score}")
            # If multiple runs exist, the plotted line reflects the best AUROC per (task, model, scorer).
            r = sub_row.sort_values("auroc", ascending=False).iloc[0]
            ys.append(float(r["auroc"]))
            los.append(float(r["ci95_lo"]))
            his.append(float(r["ci95_hi"]))
        x = np.arange(len(TASK_ORDER_STORY), dtype=float)
        ax.plot(x, ys, marker="o", label=pretty_score(score))
        yerr = np.vstack([np.array(ys) - np.array(los), np.array(his) - np.array(ys)])
        ax.errorbar(
            x, ys, yerr=yerr,
            fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
            elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
        )

    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)
    ax.set_xticks(np.arange(len(TASK_ORDER_STORY)))
    ax.set_xticklabels([TASK_PRETTY[t] for t in TASK_ORDER_STORY], rotation=0, fontsize=int(11.5 * FONT_SCALE))
    ax.set_title(MODEL_PRETTY[model])
    ax.set_ylabel("AUROC", fontsize=int(11.5 * FONT_SCALE))
    ax.set_ylim(*AUROC_YLIM)

axes[0].legend(frameon=False, title="Scorer")
fig.suptitle("Task-format effect: scorer performance shifts from MCQ → Yes/No", y=1.12, fontsize=plt.rcParams["figure.titlesize"])
safe_savefig(fig, STORY_DIR / "fig_phase2_story_3_task_format_effect_lines.pdf", bbox_inches="tight")
plt.close(fig)

# (4) Model specialization effect (approx. CI via bounds) — add labels above CI + black errorbars
rows = []
for task in TASK_ORDER_STORY:
    for score in key_scores:
        sub_m = df_main[(df_main["task"] == task) & (df_main["model"] == "mistral") & (df_main["score"] == score)]
        sub_b = df_main[(df_main["task"] == task) & (df_main["model"] == "biomistral") & (df_main["score"] == score)]
        if sub_m.empty or sub_b.empty:
            raise KeyError(f"Missing rows for diff: task={task}, score={score}, have_mistral={not sub_m.empty}, have_biomistral={not sub_b.empty}")
        r_m = sub_m.iloc[0]
        r_b = sub_b.iloc[0]
        diff = float(r_m["auroc"]) - float(r_b["auroc"])
        # NOTE: potential issue: bounds-based CI assumes independence; interpret as a conservative approximation.
        lo = float(r_m["ci95_lo"]) - float(r_b["ci95_hi"])
        hi = float(r_m["ci95_hi"]) - float(r_b["ci95_lo"])
        rows.append({"task": task, "score": score, "diff": diff, "lo": lo, "hi": hi})

dd = pd.DataFrame(rows)

# Wider figure to avoid x-label overlap.
fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

x_labels, vals, err_low, err_high = [], [], [], []
for task in TASK_ORDER_STORY:
    for score in key_scores:
        sub_dd = dd[(dd["task"] == task) & (dd["score"] == score)]
        if sub_dd.empty:
            raise KeyError(f"Missing diff row in dd for task={task}, score={score}")
        r = sub_dd.iloc[0]
        x_labels.append(f"{TASK_PRETTY[task]}\n{pretty_score(score)}")
        vals.append(float(r["diff"]))
        err_low.append(float(r["diff"] - r["lo"]))
        err_high.append(float(r["hi"] - r["diff"]))

x = np.arange(len(vals), dtype=float)

ax.bar(x, vals, width=0.65)
ax.errorbar(
    x, vals, yerr=[err_low, err_high],
    fmt="none", capsize=ERRORBAR_CAPSIZE, ecolor="black",
    elinewidth=ERRORBAR_LINEWIDTH, capthick=ERRORBAR_CAPTHICK
)
ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)

# X labels: rotate slightly + right align to prevent overlap.
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=20, ha="right")
ax.tick_params(axis="x", pad=6)  # Minor padding improves PDF legibility for multi-line labels.

ax.set_ylabel("ΔAUROC (Mistral − BioMistral)")
ax.set_title("Model specialization effect (approx. CI via bounds)", pad=18)

# Dynamic y-limits with headroom so top label never collides/clips.
ymin = min(np.array(vals) - np.array(err_low))
ymax = max(np.array(vals) + np.array(err_high))
pad = 0.12 * (ymax - ymin + 1e-9)  # Headroom even for near-constant ranges.
ax.set_ylim(ymin - 0.15 * pad, ymax + pad)

# Slightly larger value labels only for this plot (dense x labels + small vertical range).
add_value_labels_above_ci(ax, x, vals, err_high, fmt="{:.3f}", fontsize=int(12 * FONT_SCALE), pad_frac=0.025)

safe_savefig(fig, STORY_DIR / "fig_phase2_story_4_model_diff_delta_auroc.pdf", bbox_inches="tight")
plt.close(fig)