"""
Paired ablation analysis: bootstrap Δ-metrics vs a random baseline for Phase-2 medical runs.

Inputs: per (task, model) a `*.results.jsonl` with per-example labels and scorer outputs,
plus the matching `*.manifest.bootstrap_indices.npz` containing saved bootstrap resample indices.
Outputs: (i) per-run metric table + compact summary CSV, and (ii) two PDF figures with ΔAUROC and ΔSpearman.
Method: compute bootstrap distributions for AUROC and Spearman ρ, then subtract fixed baselines (0.5 / 0.0)
to obtain paired Δ distributions and 95% percentile CIs.
Determinism: fully deterministic given the on-disk `bootstrap_indices.npz` (no RNG used here).
"""

# phase_2_medical/analysis/ablations/analyze_delta_vs_random.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Plot style (kept consistent with phase2_figures.py for paper-wide comparability)
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

    # Embed fonts for consistent PDF rendering across viewers/platforms.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "dejavuserif",
})

VALUE_LABEL_FONTSIZE = int(11 * FONT_SCALE)

ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

# Human-readable labels for paper figures/tables.
TASK_PRETTY = {"medqa": "MedQA (MCQ)", "pubmedqa": "PubMedQA (Yes/No/Maybe)"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}


SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}

# Ordering convention controls x-axis ordering + cross-figure comparability.
SCORE_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]
MAIN_SCORES = set(SCORE_ORDER)

# Random baselines used for paired deltas: AUROC vs chance (0.5), Spearman vs null correlation (0.0).
BASELINES = {"auroc": 0.5, "spearman": 0.0}

# Bootstrap NPZ keys may use short names; map result.jsonl score keys -> NPZ bootstrap key.
SCORE_TO_BOOTKEY = {
    "lntp": "lntp",
    "mtp": "mtp",
    "egh_probe_oof": "egh",
    "hidden_probe_oof": "hidden",
}


def pretty_score(k: str) -> str:
    """Return a human-friendly scorer name (fallback to the raw key)."""
    kk = str(k).lower()
    return SCORE_PRETTY.get(kk, kk)


# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """
    Save a figure, retrying with versioned filenames if the target PDF is open/locked.

    NOTE: potential issue: version suffixes (_vK) can accumulate if PDFs are repeatedly left open during reruns.
    """
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
    """
    Annotate bar heights with numeric values placed above the CI upper cap.

    NOTE: potential issue: assumes the current y-limits are final (pad computed from `ax.get_ylim()`).
    """
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    pad = pad_frac * span  # pad scales with plot span to keep consistent visual spacing

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
    """Load JSONL into a list of dict rows (skips empty lines)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def np_load_first_array(npz_path: Path):
    """
    Load the first matching array from a NPZ under common key conventions.

    NOTE: potential issue: if multiple arrays exist and none match the expected keys, the first file entry is used.
    """
    z = np.load(npz_path, allow_pickle=True)
    for k in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if k in z.files:
            return z[k]
    return z[z.files[0]]


def load_bootstrap_indices_map(boot_path: Path) -> dict:
    """
    Load ALL bootstrap index arrays from an NPZ as a lowercased-key dict.

    Keys correspond to scorers (e.g., lntp/mtp/egh/hidden) and may include
    `hidden_kept_indices` (global indices into the full dataset for the hidden scorer).
    """
    z = np.load(boot_path, allow_pickle=True)
    out = {}
    for k in z.files:
        arr = z[k]
        # Some serializers store a ragged object array; stack to a 2D (B, n) index matrix.
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack(arr, axis=0)
        out[str(k).lower()] = arr.astype(int)
    return out


def find_label_key(example: dict):
    """
    Heuristically identify the label field in a Phase-2 row.

    NOTE: potential issue: label-key precedence is heuristic and may not match the canonical Phase-2 schema
    (e.g., is_error vs label/target variants) across all run variants.
    """
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")


def extract_scores(example: dict):
    """
    Extract score fields robustly from Phase-2 result rows.

    Convention:
    - Prefer explicit top-level scorer keys (lntp/mtp/egh_probe_oof/hidden_probe_oof), even if null.
    - Preserve `hidden_probe_oof=None` as np.nan so bootstrap on `hidden_kept_indices` can subset safely.
    - Also support nested dict schemas (`scores`, `wb_scores`) for compatibility across run variants.
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

    # 3) Fallback heuristic: capture numeric fields whose key suggests a known scorer.
    # NOTE: potential issue: may pick up unintended numeric fields if their names include these substrings.
    for k, v in example.items():
        kk = str(k).lower()
        if kk in scores:
            continue
        if isinstance(v, (float, int)):
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)

    return scores


def auroc_best_direction(y: np.ndarray, s: np.ndarray):
    """
    Compute AUROC and choose a polarity that yields AUROC >= 0.5.

    Convention: if AUROC < 0.5, flip scores (s -> -s) and record direction = -1.
    """
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


def spearman_rho(s: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman ρ via pandas (handles ranking/ties; returns np.nan on undefined)."""
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
            # AUROC undefined for a single-class resample; skip this replicate (reduces effective B).
            continue
        aucs.append(roc_auc_score(yy, ss))
        rr = spearman_rho(ss, yy)
        if not np.isnan(rr):
            rhos.append(rr)

    aucs = np.asarray(aucs, dtype=float)
    rhos = np.asarray(rhos, dtype=float)
    return aucs, rhos


def ci_from_dist(dist: np.ndarray, alpha=0.05):
    """Return (mean, lo, hi) using percentile CI on a bootstrap distribution (NaN triplet if empty)."""
    if dist.size == 0:
        # NOTE: potential issue: upstream skips (e.g., single-class resamples) can yield empty distributions.
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

# CSVs sollen ebenfalls hier gespeichert werden (co-locate artifacts for this ablation).
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
        # Convention: only analyze runs produced with explicit bootstrap size marker.
        continue

    # Parse task/model from prefix "<task>_<model>" (lowercased for stable joins/labels).
    prefix = name.split(".B")[0]
    if "_" not in prefix:
        continue
    task, model = prefix.split("_", 1)
    task = task.lower()
    model = model.lower()

    # Bootstrap indices: try canonical naming first, then fall back to a glob match.
    stem = name.replace(".results.jsonl", "")
    boot_path = FINAL_DIR / f"{stem}.manifest.bootstrap_indices.npz"
    if not boot_path.exists():
        gl = sorted(FINAL_DIR.glob(f"{stem}*bootstrap_indices*.npz"))
        boot_path = gl[0] if gl else None

    if boot_path is None or not boot_path.exists():
        # Missing indices => cannot reproduce the intended bootstrap; skip run to avoid mixing schemes.
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
        # No rows => nothing to analyze; do not emit partial records.
        continue

    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # Align per-example scorer arrays by row order; later we intersect keys to keep common scorers only.
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

    # Optional: for the hidden scorer, bootstrap indices are defined over a kept-subset.
    hidden_kept = boot_map.get("hidden_kept_indices", None)

    for score_key, s_raw in S.items():
        score_l = score_key.lower()   # normalize once; used for mapping + special-casing
        s_raw = np.asarray(s_raw, dtype=float)

        # Determine score polarity once per scorer to enforce AUROC >= 0.5 (and reuse across all bootstraps).
        # For hidden: direction must be learned on the kept subset (same population as the hidden bootstrap).
        if score_l == "hidden_probe_oof" and ("hidden_kept_indices" in boot_map):
            hk = boot_map["hidden_kept_indices"]
            au_full, direction = auroc_best_direction(y[hk], s_raw[hk])
        else:
            au_full, direction = auroc_best_direction(y, s_raw)

        # Apply chosen polarity consistently to all downstream metrics (AUROC + Spearman).
        s = s_raw * direction

        # Pick correct bootstrap indices for this scorer (NPZ keys may use shortened names).
        boot_key = SCORE_TO_BOOTKEY.get(score_l, score_l)

        if boot_key not in boot_map:
            print(f"[WARN] No bootstrap indices for score_key={score_key} in {boot_path.name}; skipping.")
            continue

        boot_idx = boot_map[boot_key]

        # Special case: hidden is bootstrapped on the kept-subset (indices are relative to that subset).
        if score_l == "hidden_probe_oof":
            if hidden_kept is None:
                raise KeyError(
                    f"hidden_kept_indices missing in {boot_path.name} but required for hidden bootstrap."
                )
            # hidden_kept are indices into FULL y/s arrays (subset selection happens before indexing by boot_idx).
            y_use = y[hidden_kept]
            s_use = s[hidden_kept]
        else:
            y_use = y
            s_use = s

        au_dist, sp_dist = bootstrap_metric_distributions(y_use, s_use, boot_idx)

        # Paired deltas vs fixed random baselines (computed in bootstrap-space).
        d_au = au_dist - BASELINES["auroc"]
        d_sp = sp_dist - BASELINES["spearman"]

        # CI in delta-space (paired): preserves within-replicate correlation between metric and baseline.
        d_au_mean, d_au_lo, d_au_hi = ci_from_dist(d_au, alpha=0.05)
        d_sp_mean, d_sp_lo, d_sp_hi = ci_from_dist(d_sp, alpha=0.05)

        # Also store absolute metrics for reference (helps sanity-check direction/baseline effects).
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
            # Significance heuristic: Δ CI entirely above 0 implies improvement vs random baseline.
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



# Summary (compact, paper/table-friendly slice)
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
# Stable ordering: prefer canonical task/model order, then append any additional discovered categories.
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])

score_order = [k for k in SCORE_ORDER if k in set(df["score_key"])]
score_order += sorted([k for k in set(df["score_key"]) if k not in score_order])


def plot_delta(metric: str, outpath: Path):
    """Render a task×model grid for a given Δ-metric with 95% CI error bars."""
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

    # Global y-limits: enforce a shared scale across subplots to support visual comparison.
    vals = df[[lo_col, hi_col]].to_numpy(dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # Fallback range to keep plots readable if all distributions are empty/NaN.
        y_min, y_max = -0.05, 0.20
    else:
        pad = 0.01 * (float(np.max(vals)) - float(np.min(vals)) + 1e-9)
        y_min = float(np.min(vals)) - pad
        y_max = float(np.max(vals)) + pad
        # Ensure the Δ=0 reference line is visible even if all CIs are positive/negative.
        y_min = min(y_min, -0.01)
        y_max = max(y_max, 0.01)
        # Add headroom so value labels do not collide with subplot titles (keeps scale comparable).
        y_max = y_max + 0.10 * (y_max - y_min + 1e-9)


    fig, axes = plt.subplots(
        len(models), len(tasks),
        figsize=(5.6 * len(tasks), 3.9 * len(models)),
        sharey=True
    )

    # Normalize axes shape to 2D for uniform indexing across 1×N / N×1 / 1×1 layouts.
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

            # Enforce consistent scorer ordering; missing scorers become NaN via reindex below.
            sub["score_key"] = pd.Categorical(sub["score_key"], categories=score_order, ordered=True)
            sub = sub.sort_values("score_key")

            x = np.arange(len(score_order), dtype=float)

            # Alignment invariant: reindex(score_order) ensures y/lo/hi arrays share identical ordering/length.
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
            ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # Δ=0: no improvement vs baseline

            ax.set_xticks(x)
            ax.set_xticklabels([pretty_score(k) for k in score_order], rotation=20, ha="right")

            ax.set_ylim(y_min, y_max)
            if c == 0:
                ax.set_ylabel(ylabel)

            if r == 0:
                ax.set_title(TASK_PRETTY.get(task, task), pad=13) 

            if c == len(tasks) - 1:
                # Row label on the rightmost subplot to avoid duplicating labels in every panel.
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