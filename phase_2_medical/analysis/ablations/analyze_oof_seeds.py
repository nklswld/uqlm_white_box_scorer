# phase_2_medical/analysis/ablations/analyze_oof_seeds.py
#
# Ablation: OOF Robustness over different random seeds (e.g., 0, 42, 123, ...)
# Goal: Probe performance should not be “seed-lucky”.
#
# Outputs:
#  - phase_2_medical/outputs/ablations/oof_seeds/analysis_oof_seeds_metrics.csv
#      (per-seed metrics with bootstrap CI, for LNTP/MTP/EGH/Hidden)
#  - phase_2_medical/outputs/ablations/oof_seeds/analysis_oof_seeds_summary.csv
#      (mean/std across seeds, per task×model×score)
#  - phase_2_medical/outputs/figs/ablations/oof_seeds/
#      fig_ablation_oof_seeds_auroc_overlay.pdf
#      fig_ablation_oof_seeds_spearman_overlay.pdf

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: Consistent with phase2_figures.py  
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

AUROC_YLIM = (0.45, 0.85)
SPEARMAN_YLIM = (-0.10, 0.70)

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

# Expected ablation folder
ABL_DIR = ROOT / "outputs" / "ablations" / "oof_seeds"
FIGS_DIR = ROOT / "outputs" / "figs" / "ablations" / "oof_seeds"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
ABL_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = FIGS_DIR / "analysis_oof_seeds_metrics.csv"
OUT_CSV_SUMMARY = FIGS_DIR / "analysis_oof_seeds_summary.csv"


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
# Helpers (consistent with other ablation scripts)
# ============================================================
def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_bootstrap_npz(npz_path: Path) -> np.lib.npyio.NpzFile:
    """Open bootstrap NPZ (caller should close)."""
    return np.load(npz_path, allow_pickle=True)


def _stack_if_object(arr: np.ndarray) -> np.ndarray:
    """Stack object-dtype bootstrap index arrays to (B, N)."""
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr, axis=0)
    return arr.astype(int)


def get_bootstrap_indices_for_score(z: np.lib.npyio.NpzFile, score_key: str) -> np.ndarray:
    """
    Retrieve the correct bootstrap index matrix for a given score key.
    Expected keys in NPZ often include: lntp, mtp, egh, egh_ge, hidden, ...
    """
    sk = str(score_key).lower()

    # Map result-score keys -> NPZ keys (runner convention)
    key_map = {
        "lntp": "lntp",
        "mtp": "mtp",
        "egh_probe_oof": "egh",          # common stored key
        "egh_probe_ge": "egh_ge",        # if present
        "hidden_probe_oof": "hidden",
    }

    # Try mapped key first
    cand = key_map.get(sk, sk)
    if cand in z.files:
        return _stack_if_object(z[cand])

    # Fallback: try a few common variants
    for alt in [sk, sk.replace("_probe_oof", ""), sk.replace("_probe", ""), "indices", "boot_idx", "bootstrap_indices", "idx"]:
        if alt in z.files:
            return _stack_if_object(z[alt])

    # Final fallback: first array (best-effort) – but warn loudly
    first = z.files[0]
    print(f"[WARN] No bootstrap key for score='{sk}' found. Falling back to first NPZ entry: '{first}'")
    return _stack_if_object(z[first])


def get_hidden_kept_indices(z: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Return hidden_kept_indices if present (maps kept subset back to full indices)."""
    for k in ["hidden_kept_indices", "kept_indices", "hidden_kept_full_indices"]:
        if k in z.files:
            return np.asarray(z[k], dtype=int)
    return None


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
    """
    Pick score polarity so AUROC >= 0.5 by flipping sign if needed.
    Returns (auroc, direction) with direction in {+1, -1}.
    """
    au = roc_auc_score(y, s_raw)
    if au < 0.5:
        return roc_auc_score(y, -s_raw), -1.0
    return au, +1.0


def bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05):
    """
    Compute bootstrap mean and (1-alpha) CI using *provided* bootstrap indices (exact reproducibility).
    boot_idx: shape (B, N) indices into y/s arrays.
    """
    vals = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # handle degenerate resamples with single class
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
    """
    Bootstrap CI for Spearman rho using provided bootstrap indices.
    """
    vals = []
    y = np.asarray(y)
    s = np.asarray(s)
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        if len(np.unique(yy)) < 2:
            continue
        # Spearman via pandas (robust + consistent with your other scripts)
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
    # task
    task = manifest.get("task", None)
    if task is None:
        # try filename
        name = fallback_path.name.lower()
        if "medqa" in name:
            task = "medqa"
        elif "pubmedqa" in name:
            task = "pubmedqa"
    task = (task or "unknown").lower()

    # model
    model_name = manifest.get("model_name") or manifest.get("model") or ""
    model = None
    mn = str(model_name).lower()
    if "biomistral" in mn:
        model = "biomistral"
    elif "mistral" in mn:
        model = "mistral"

    if model is None:
        # try filename
        name = fallback_path.name.lower()
        if "biomistral" in name:
            model = "biomistral"
        elif "mistral" in name:
            model = "mistral"

    model = model or "unknown"
    return task, model


def infer_seed(manifest: dict, manifest_path: Path):
    # Prefer manifest seed if present
    for key in ["seed", "random_seed"]:
        if key in manifest:
            try:
                return int(manifest[key])
            except Exception:
                pass

    # Try filename / parents: seed_42, seed42, etc.
    txt = str(manifest_path)
    m = re.search(r"seed[_\-]?(\d+)", txt.lower())
    if m:
        return int(m.group(1))

    # Fallback
    return -1


def find_runs(abl_root: Path):
    """
    We expect per-seed runs to produce:
      *.manifest.json
      corresponding *.results.jsonl
      corresponding *.manifest.bootstrap_indices.npz
    """
    manifests = sorted(abl_root.rglob("*.manifest.json"))
    runs = []
    for mp in manifests:
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue

        task, model = infer_task_model_from_manifest(manifest, mp)
        seed = infer_seed(manifest, mp)

        # Best-effort locate results + bootstrap indices next to manifest
        # Common naming convention:
        #   X.manifest.json
        #   X.results.jsonl
        #   X.manifest.bootstrap_indices.npz
        stem = mp.name.replace(".manifest.json", "")
        candidates_results = [
            mp.with_name(f"{stem}.results.jsonl"),
            mp.with_name(f"{stem}.results.json"),
            mp.with_name(f"{stem}.results.jsonl.gz"),
        ]
        results_path = None
        for c in candidates_results:
            if c.exists():
                results_path = c
                break

        # fallback: any *.results.jsonl in same dir that shares prefix
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
        boot_path = None
        for c in candidates_boot:
            if c.exists():
                boot_path = c
                break

        # fallback: look for any *bootstrap_indices*.npz in the same directory
        if boot_path is None:
            gl = sorted(mp.parent.glob("*bootstrap_indices*.npz"))
            if gl:
                boot_path = gl[0]

        if results_path is None or boot_path is None:
            continue

        runs.append((task, model, seed, mp, results_path, boot_path))

    return runs


# ============================================================
# Collect per-seed metrics
# ============================================================
runs = find_runs(ABL_DIR)
print("Found runs:", len(runs))
if len(runs) == 0:
    sample_listing = sorted([p.name for p in ABL_DIR.glob("**/*")][:200])
    raise RuntimeError(
        "No runs found under outputs/ablations/oof_seeds.\n"
        "Sample listing:\n" + "\n".join(sample_listing)
    )

records = []
for task, model, seed, manifest_path, results_path, boot_path in runs:
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

    S = {k: np.array([d.get(k, np.nan) for d in score_dicts], dtype=float) for k in keys}
    # keep only main scores for this ablation
    S = {k: v for k, v in S.items() if k in MAIN_SCORES}

    
    with load_bootstrap_npz(boot_path) as z:
        kept_hidden = get_hidden_kept_indices(z)

        for score_key, s_raw in S.items():
            sk = str(score_key).lower()

            # --- Select correct bootstrap indices for this score ---
            boot_idx = get_bootstrap_indices_for_score(z, sk)

            # --- Align yy/ss to the bootstrap index length ---
            if sk == "hidden_probe_oof":
                # Hidden is computed only on kept subset; NPZ should contain kept indices mapping.
                if kept_hidden is None:
                    raise KeyError(
                        f"NPZ missing hidden_kept_indices but score '{sk}' requires them: {boot_path}"
                    )
                yy = y[kept_hidden]
                ss = s_raw[kept_hidden]
                # Hidden scores should be finite on kept subset; if not, something is inconsistent.
                if not np.isfinite(ss).all():
                    raise ValueError(
                        f"Hidden scores still contain NaN/inf after applying kept indices. "
                        f"Run={results_path}"
                    )
            else:
                # For non-hidden scores we expect full coverage (no NaNs).
                if not np.isfinite(s_raw).all():
                    # deterministic + correct bootstrap requires stored indices for the same N.
                    # Safer to SKIP than to silently remap indices (changes the resampling scheme).
                    print(f"[WARN] Non-hidden score '{sk}' has NaNs; skipping for exact bootstrap reproducibility: {results_path}")
                    continue
                yy = y
                ss = s_raw

            # --- Polarity convention + metrics ---
            au, direction = auroc_with_best_direction(yy, ss)
            s = ss * direction

            au_mean, au_lo, au_hi = bootstrap_ci_from_indices(yy, s, boot_idx, alpha=0.05)
            sp_mean, sp_lo, sp_hi = bootstrap_spearman_ci_from_indices(yy, s, boot_idx, alpha=0.05)

            records.append({
                "task": task,
                "model": model,
                "seed": int(seed),
                "score_key": sk,
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
df = df.sort_values(["task", "model", "score_key", "seed"]).reset_index(drop=True)

df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)

if df.empty:
    raise RuntimeError("No rows computed. Check that results contain the expected MAIN_SCORES keys.")


# ============================================================
# Summary: mean/std across seeds per task×model×score
# (This matches your ablation-table definition: mean/std over seeds)
# ============================================================
g = df.groupby(["task", "model", "score_key"], as_index=False)
df_sum = g.agg(
    seeds=("seed", "nunique"),
    auroc_mean_over_seeds=("auroc_boot_mean", "mean"),
    auroc_std_over_seeds=("auroc_boot_mean", "std"),
    spearman_mean_over_seeds=("spearman_rho_boot_mean", "mean"),
    spearman_std_over_seeds=("spearman_rho_boot_mean", "std"),
)

# If only one seed, std becomes NaN; replace with 0 for cleaner plots/tables
for c in ["auroc_std_over_seeds", "spearman_std_over_seeds"]:
    df_sum[c] = df_sum[c].fillna(0.0)

df_sum.to_csv(OUT_CSV_SUMMARY, index=False)
print("Wrote:", OUT_CSV_SUMMARY)


# ============================================================
# Plotting: overlay (1 panel per task, lines = models, x = score)
# Here: y = mean_over_seeds, errorbars = std_over_seeds (requested)
# ============================================================
tasks = [t for t in ["medqa", "pubmedqa"] if t in set(df_sum["task"])]
tasks += sorted([t for t in set(df_sum["task"]) if t not in tasks])

models = [m for m in ["mistral", "biomistral"] if m in set(df_sum["model"])]
models += sorted([m for m in set(df_sum["model"]) if m not in models])

score_order = [k for k in MAIN_SCORES if k in set(df_sum["score_key"])]
score_order += sorted([k for k in set(df_sum["score_key"]) if k not in score_order])

x = np.arange(len(score_order), dtype=float)


def plot_overlay(metric: str, y_lim, title: str, outpath: Path):
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.2 * len(tasks), 4.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        for model in models:
            sub = df_sum[(df_sum["task"] == task) & (df_sum["model"] == model)].copy()
            if sub.empty:
                continue
            sub = sub.set_index("score_key").reindex(score_order).reset_index()

            if metric == "auroc":
                yv = sub["auroc_mean_over_seeds"].to_numpy(dtype=float)
                yerr = sub["auroc_std_over_seeds"].to_numpy(dtype=float)
                ylabel = "AUROC (mean over seeds)"
                hline = 0.5
            else:
                yv = sub["spearman_mean_over_seeds"].to_numpy(dtype=float)
                yerr = sub["spearman_std_over_seeds"].to_numpy(dtype=float)
                ylabel = "Spearman ρ (mean over seeds)"
                hline = 0.0

            # line + markers
            ax.plot(x, yv, marker="o", label=MODEL_PRETTY.get(model, model))

            # errorbars in black for contrast
            ax.errorbar(x, yv, yerr=yerr, fmt="none", capsize=3, ecolor="black")

            # value labels slightly above errorbar
            add_pad_up = 0.017 * (y_lim[1] - y_lim[0])
            add_pad_dn = 0.017 * (y_lim[1] - y_lim[0])

            for xi, score_key, yi, ei in zip(x, score_order, yv, yerr):
                if not np.isfinite(yi):
                    continue

                err = (ei if np.isfinite(ei) else 0.0)

                # Regel: bestimmte Labels "unter" den Punkt setzen
                place_below = (
                    (task == "medqa" and model == "biomistral" and score_key in ["lntp", "mtp"]) or
                    (task == "pubmedqa" and model == "biomistral" and score_key == "hidden_probe_oof")
                )

                if place_below:
                    y_text = yi - err - add_pad_dn
                    va = "top"
                else:
                    y_text = yi + err + add_pad_up
                    va = "bottom"

                ax.text(
                    xi, y_text, f"{yi:.3f}",
                    ha="center", va=va, fontsize=VALUE_LABEL_FONTSIZE
                )

        ax.axhline(hline, linestyle="--", linewidth=1)
        ax.set_ylim(*y_lim)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_score(k) for k in score_order], rotation=0)
        ax.set_xlabel("Scorer")
        ax.set_title(TASK_PRETTY.get(task, task))
        ax.grid(False)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, title="Model")

    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)

plot_overlay(
    metric="auroc",
    y_lim=AUROC_YLIM,
    title="OOF Robustness (Seeds) — AUROC mean ± std over seeds",
    outpath=FIGS_DIR / "fig_ablation_oof_seeds_auroc_overlay.pdf",
)

plot_overlay(
    metric="spearman",
    y_lim=SPEARMAN_YLIM,
    title="OOF Robustness (Seeds) — Spearman ρ mean ± std over seeds",
    outpath=FIGS_DIR / "fig_ablation_oof_seeds_spearman_overlay.pdf",
)

print("Done. Figures in:", FIGS_DIR)