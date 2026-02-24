"""
Analyze a bootstrap-budget ablation by quantifying confidence-interval (CI) width convergence as the
number of bootstrap resamples B increases (e.g., B ∈ {1000, 2000, 5000, 10000}).
Inputs: per-run *.manifest.json, paired results (*.results.jsonl/.json), and persisted bootstrap indices
(*bootstrap_indices*.npz) under outputs/ablations/bootstrap_budget/.
Outputs: per-run metric table (CSV), aggregated CI half-width summary (CSV), and PDF figures.
Reproducibility: CI estimates are deterministic given the stored bootstrap indices and input artifacts.
"""

# phase_2_medical/analysis/ablations/analyze_bootstrap_budget.py
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
FONT_SCALE = 1.5

# Global Matplotlib defaults for consistent typography across repository figures.
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

    # Embed fonts to avoid PDF text rendering differences across platforms/viewers.
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

# Canonical scorer keys expected in results; analysis intentionally restricts to these (repo convention).
SCORE_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}
SCORE_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]
MAIN_SCORES = set(SCORE_ORDER)


def pretty_score(score_key: str) -> str:
    """Map internal scorer keys to display labels used in figures."""
    k = str(score_key).lower()
    return SCORE_PRETTY.get(k, k)


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical

ABL_DIR = ROOT / "outputs" / "ablations" / "bootstrap_budget"
FIGS_DIR = ROOT / "outputs" / "figures_tables" / "ablations" / "bootstrap_budget"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
ABL_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: potential issue: metrics CSVs are written under FIGS_DIR (not ABL_DIR); keep paths stable for downstream tooling.
OUT_CSV = FIGS_DIR / "analysis_bootstrap_budget_metrics.csv"
OUT_CSV_SUMMARY = FIGS_DIR / "analysis_bootstrap_budget_summary.csv"


# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a figure; if the target is locked (common on Windows), write a versioned fallback PDF."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        # Common on Windows when the PDF is open in a viewer; scan deterministic suffixes _v2.._v49.
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
    """Load a JSONL file into a list of dicts; ignores blank lines for robustness."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# Mapping from score_key (results schema) to NPZ array key (bootstrap indices schema).
SCORE_TO_BOOTKEY = {
    "lntp": "lntp",
    "mtp": "mtp",
    "egh_probe_oof": "egh_ge",
    "hidden_probe_oof": "hidden",
}

def load_npz_array(npz_path: Path, key: str) -> np.ndarray:
    """Load an array from an NPZ and normalize object-dtype arrays into a stacked numeric ndarray."""
    z = np.load(npz_path, allow_pickle=True)
    if key not in z.files:
        raise KeyError(f"[BOOT] key='{key}' not in {npz_path.name}. Available: {list(z.files)}")
    arr = z[key]
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # Some pipelines persist per-resample arrays as dtype=object; stack into shape (B, n).
        arr = np.stack(arr, axis=0)
    return arr

def load_bootstrap_indices(boot_path: Path, score_key: str) -> np.ndarray:
    """Load precomputed bootstrap resample indices for a given scorer (determinism anchor)."""
    npz_key = SCORE_TO_BOOTKEY.get(score_key, score_key)
    return load_npz_array(boot_path, npz_key).astype(int)

def load_hidden_kept_indices(boot_path: Path) -> np.ndarray | None:
    """Load kept-example indices for 'hidden' scorer, if it was computed on a filtered subset."""
    z = np.load(boot_path, allow_pickle=True)
    if "hidden_kept_indices" in z.files:
        return z["hidden_kept_indices"].astype(int)
    return None


def find_label_key(example: dict):
    """Identify the binary label field using a fixed preference order (repo-wide convention)."""
    # NOTE: potential issue: relies on a small set of expected keys; new result schemas must extend this list.
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")


def extract_scores(example: dict):
    """Extract scorer outputs from a result record using schema-aware keys with a numeric-field fallback."""
    # Preferred schemas: nested scorer dicts.
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    # Fallback schema: accept numeric top-level fields whose names suggest known scorers.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            # Heuristic: keep only numeric fields matching known scorer identifiers.
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)
    return scores


def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and a polarity (+1/-1) such that the oriented AUROC is ≥ 0.5."""
    # Convention: orient scores so that larger values correspond to the positive label (y=1).
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Percentile bootstrap CI for AUROC using persisted resample indices; skips degenerate resamples."""
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: AUROC undefined when only one class is present.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))
    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        # NOTE: potential issue: all resamples degenerate implies unstable CI estimate; propagate NaNs.
        return np.nan, np.nan, np.nan
    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi


def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Percentile bootstrap CI for Spearman ρ using persisted resample indices; skips NaN/degenerate resamples."""
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Degenerate resample: correlation/AUROC-style summaries are unstable when y is constant.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        # Correlation may be NaN for constant vectors or insufficient variability.
        if pd.isna(rho):
            continue
        rhos.append(float(rho))
    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0:
        # NOTE: potential issue: all resamples degenerate/NaN implies unstable CI estimate; propagate NaNs.
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
    """Infer (task, model) from the folder naming convention '<task>_<model>' above the B_* directory."""
    # Expected layout: .../bootstrap_budget/medqa_biomistral/B_1000/<file>.manifest.json
    try:
        task_model = manifest_path.parent.parent.name.lower()
        parts = task_model.split("_")
        task = parts[0]
        model = parts[1] if len(parts) > 1 else "unknown"
        return task, model
    except Exception:
        # Silent fallback keeps discovery robust to unexpected directory layouts.
        return "unknown", "unknown"


def parse_B(manifest_path: Path, manifest: dict):
    """Resolve bootstrap budget B from manifest fields or path conventions; returns -1 if unknown."""
    # Prefer explicit manifest metadata, falling back to folder/filename patterns for legacy runs.
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
    """Locate paired results (JSONL/JSON) and persisted bootstrap indices (NPZ) for a manifest."""
    stem = manifest_path.name.replace(".manifest.json", "")

    # Results are expected as JSONL; tolerate legacy JSON dumps as a fallback.
    results_path = manifest_path.with_name(stem + ".results.jsonl")
    if not results_path.exists():
        alt = manifest_path.with_name(stem + ".results.json")
        results_path = alt if alt.exists() else None

    # Bootstrap indices: prefer the canonical stem; otherwise take the first matching NPZ in the folder.
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

    # Label extraction: decide label key once (first record) and apply consistently for the full run.
    y_key = find_label_key(rows[0])
    y = np.array([int(r[y_key]) for r in rows], dtype=int)

    # Score extraction: intersect per-row keys to enforce that every scorer has a value for every example.
    score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]
    keys = set(score_dicts[0].keys())
    for d in score_dicts[1:]:
        keys &= set(d.keys())
    keys = sorted([str(k).lower() for k in keys])

    # Restrict to canonical scorers to avoid accidentally including incidental numeric fields.
    keys = [k for k in keys if k in MAIN_SCORES]
    if not keys:
        continue

    # Invariant: each score array has shape (N,) aligned to y by construction (same row order).
    S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}

    for score_key, s_raw in S.items():
        # Use score-specific persisted indices to keep CI estimates deterministic across reruns.
        boot_idx = load_bootstrap_indices(boot_path, score_key)

        # Hidden scorer may be computed on a subset; align y/s to that subset (or drop non-finite scores).
        y_use = y
        s_use = s_raw
        if score_key == "hidden_probe_oof":
            kept = load_hidden_kept_indices(boot_path)
            if kept is not None:
                y_use = y[kept]
                s_use = s_raw[kept]
            else:
                # Fallback if no kept indices stored: drop NaNs/inf with a shared mask to preserve alignment.
                m = np.isfinite(s_raw)
                y_use = y[m]
                s_use = s_raw[m]

        # Polarity convention: flip sign if needed so that AUROC ≥ 0.5 (higher score => more positive label).
        _, direction = auroc_with_best_direction(y_use, s_use)
        s = s_use * direction

        au_mean, au_lo, au_hi = bootstrap_ci_from_indices(y_use, s, boot_idx, alpha=0.05)
        sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(y_use, s, boot_idx, alpha=0.05)

        records.append({
            "task": task,
            "model": model,
            "B": int(B),
            "score_key": score_key,
            "direction": float(direction),
            "N": int(len(y_use)),
            "pos_rate": float(y_use.mean()) if len(y_use) > 0 else np.nan,

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
# Aggregation is by (score_key, B): each row summarizes CI half-width across all discovered runs.
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
# Deterministic ordering: prefer canonical task/model/score sequences, then append any extras discovered on disk.
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df["task"])]
tasks += sorted([t for t in set(df["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df["model"])]
models += sorted([m for m in set(df["model"]) if m not in models])

score_order = [k for k in SCORE_ORDER if k in set(df["score_key"])]
score_order += sorted([k for k in set(df["score_key"]) if k not in score_order])

# X-axis budgets: only positive B values are treated as valid experimental points.
Bs = sorted([int(x) for x in set(df["B"]) if int(x) > 0])
xpos = np.arange(len(Bs), dtype=float)


# ============================================================
# Option A: CI half-width vs B with tight global y-axis
# ============================================================
def plot_ciwidth(metric: str, title: str, outpath: Path):
    """Plot CI half-width vs B per task×model panel with shared y-limits for cross-panel comparability."""
    # Column selection by metric family (AUROC vs Spearman ρ).
    if metric == "auroc":
        col = "auroc_ci95_halfwidth"
        ylabel = "95% CI half-width (AUROC)"
    else:
        col = "spearman_ci95_halfwidth"
        ylabel = "95% CI half-width (Spearman ρ)"

    # -------- Option A: global tight y-limits (same across panels) --------
    # Design choice: a shared tight y-range enables direct visual comparison across tasks/models.
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

    # Grid layout: rows=models, cols=tasks (consistent panel semantics across figures).
    fig, axes = plt.subplots(
        len(models), len(tasks),
        figsize=(5.3 * len(tasks), 3.7 * len(models)),
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

                # Reindex to global Bs so missing budgets render as gaps (NaN) without shifting x positions.
                sub = sub.set_index("B").reindex(Bs).reset_index()
                yv = sub[col].to_numpy(dtype=float)

                ax.plot(
                    xpos, yv,
                    marker="o",
                    linewidth=2.0,
                    markersize=6,
                    label=pretty_score(score_key),
                )

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

    # Legend is extracted from the first panel; ordering is controlled by score_order above.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=min(4, len(score_order)),
        frameon=False,
        title="Scorer"
    )
    fig.suptitle(title, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.42)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", outpath)


# ============================================================
# Option B: Δ CI half-width vs B=max (typically 10000)
# ============================================================
def plot_delta_ci(metric: str, title: str, outpath: Path):
    """Plot Δ(CI half-width) vs B=max per task×model panel with shared y-limits (gaps indicate missing refs)."""
    if metric == "auroc":
        col = "auroc_ci95_halfwidth"
        ylabel = "Δ 95% CI half-width\n(AUROC) (vs B=max)"
    else:
        col = "spearman_ci95_halfwidth"
        ylabel = "Δ 95% CI half-width\n(Spearman ρ) (vs B=max)"

    # Reference is defined as the largest observed B across discovered runs (data-driven B_ref).
    B_ref = max(Bs) if Bs else None
    if B_ref is None:
        raise RuntimeError("No bootstrap budgets found (Bs empty).")

    # Compute delta per (task, model, score_key): y(B) - y(B_ref) after a left join on reference rows.
    df_delta = df[["task", "model", "score_key", "B", col]].copy()
    ref = df_delta[df_delta["B"] == B_ref][["task", "model", "score_key", col]].rename(columns={col: "ref"})
    # NOTE: potential issue: missing ref rows yield NaN deltas (left join); plots will show gaps without warning.
    df_delta = df_delta.merge(ref, on=["task", "model", "score_key"], how="left")
    df_delta["delta"] = df_delta[col] - df_delta["ref"]

    # Tight global y-limits around deltas (same across panels) for consistent interpretation.
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
        figsize=(5.3 * len(tasks), 3.7 * len(models)),
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
                # Align to global Bs to preserve identical x-coordinates across panels/scorers.
                sub = sub.set_index("B").reindex(Bs).reset_index()
                yv = sub["delta"].to_numpy(dtype=float)
                ax.plot(xpos, yv, marker="o", label=pretty_score(score_key))

            # Baseline at zero highlights convergence to the B_ref CI half-width.
            ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)
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
    fig.suptitle(title, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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