"""
Analyze token-score length bias across ablation runs and generate appendix-ready figures/tables.
Inputs: per-run `token_bias.manifest.json` plus `token_bias.results.jsonl` (or `results.jsonl` fallback),
and optionally `k_sweep.manifest.json` or an embedded `answer_span_k_sweep` block in the token-bias manifest.
Outputs: CSVs with AUROC + Spearman(length dependence) metrics and PDF plots (AUROC bars, Spearman bars, k-sweep lines).
Direction handling: scores may be sign-flipped per metric to ensure AUROC >= 0.5 (reported via `direction`).
Determinism: fully deterministic given fixed input files; no stochastic resampling/bootstrapping is used here.
"""

# phase_2_medical/analysis/ablations/analyze_token_score_bias.py
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score


# ============================================================
# Style: consistent with phase2_figures.py
# ============================================================
FONT_SCALE = 1.5  # Global typographic scale; keep consistent across appendix figures.

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

# ---------------------------------------------------------------------
# Plot styling knobs (global, for print/readability)
# ---------------------------------------------------------------------
# NOTE: potential issue: ERRORBAR_* constants are defined for consistency with other scripts,
# but no confidence intervals are plotted in this analysis (point estimates only).
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.6
ERRORBAR_CAPTHICK = 1.6
BASELINE_LINEWIDTH = 1.4

# ============================================================
# Robust save helper (Windows PDF file lock)
# ============================================================
def safe_savefig(fig, outpath: Path, **kwargs):
    """Save a PDF robustly by writing a temp file then atomically replacing the target (with retries)."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: potential issue: temp file is created next to the target; cross-filesystem moves are not supported.
    tmp = outpath.with_name(outpath.stem + ".__tmp__" + outpath.suffix)

    # Enforce PDF output regardless of suffix to avoid backend/format ambiguity.
    fig.savefig(tmp, format="pdf", **kwargs)

    # Atomically replace with retry to tolerate transient Windows file locks.
    for _ in range(15):
        try:
            tmp.replace(outpath)
            return outpath
        except PermissionError:
            time.sleep(0.2)

    # Fallback: write versioned filename if the canonical output remains locked.
    stem, suffix = outpath.stem, outpath.suffix
    for k in range(2, 50):
        alt = outpath.with_name(f"{stem}_v{k}{suffix}")
        try:
            tmp.replace(alt)
            print(f"[WARN] Permission denied for {outpath.name}. Wrote: {alt.name}")
            return alt
        except PermissionError:
            time.sleep(0.2)

    raise PermissionError(f"Could not write {outpath} (still locked).")

# ============================================================
# Helper: labels above bars (no CI)
# ============================================================
def add_value_labels(ax, x_positions, y_values, fmt="{:.3f}", fontsize=None,
                     pos_offset=6, neg_offset=-8):
    """Annotate bar heights; negative values get labels below the bar."""
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    for x, y in zip(x_positions, y_values):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue

        y = float(y)
        if y >= 0:
            xytext = (0, pos_offset)
            va = "bottom"
        else:
            xytext = (0, neg_offset)
            va = "top"

        ax.annotate(
            fmt.format(y),
            (float(x), y),
            textcoords="offset points",
            xytext=xytext,
            ha="center",
            va=va,
            fontsize=fontsize,
            clip_on=False,
        )


# ============================================================
# IO helpers
# ============================================================
def load_json(path: Path):
    """Load a UTF-8 JSON file into a Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path):
    """Load JSONL (one JSON object per non-empty line) into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# Metrics helpers
# ============================================================
def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and a sign (±1) that makes AUROC >= 0.5 via score inversion if needed."""
    y = np.asarray(y).reshape(-1)
    s = np.asarray(s).reshape(-1)

    # Guard 1: AUROC undefined if only one class present (cannot rank positives vs negatives).
    if np.unique(y).size < 2:
        return np.nan, +1.0

    # Guard 2: near-constant scores imply no meaningful ranking (and may trigger metric edge cases).
    if float(np.nanstd(s)) < 1e-12:
        return np.nan, +1.0

    # Convention: if AUROC < 0.5, flip scores so "higher is better" for downstream correlations/plots.
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return float(au), +1.0

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical (repo-local anchor; avoids CWD dependence)
ABL_ROOT = ROOT / "outputs" / "ablations" / "token_score_bias"
OUT_DIR = ROOT / "outputs" / "figures_tables" / "ablations" / "token_score_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("ROOT =", ROOT)
print("ABL_ROOT =", ABL_ROOT)
print("ABL_ROOT exists =", ABL_ROOT.exists())
print("OUT_DIR =", OUT_DIR)

if not ABL_ROOT.exists():
    raise FileNotFoundError(f"Token score bias folder not found: {ABL_ROOT}")


# ============================================================
# Locate required files
# - token_bias.manifest.json
# - token_bias.results.jsonl
# - k_sweep.manifest.json (optional; may be duplicated inside token_bias.manifest.json)
# ============================================================
token_bias_manifest_paths = sorted(ABL_ROOT.glob("**/token_bias.manifest.json"))
k_sweep_manifest_paths = sorted(ABL_ROOT.glob("**/k_sweep.manifest.json"))
# NOTE: potential issue: k_sweep_manifest_paths is currently not used directly; 
# k-sweep files are resolved per run via ks_path below.

if len(token_bias_manifest_paths) == 0:
    listing = sorted([str(p.relative_to(ABL_ROOT)) for p in ABL_ROOT.glob("**/*")][:250])
    raise FileNotFoundError("No token_bias.manifest.json found. Sample listing:\n" + "\n".join(listing))

# If multiple are present, analyze all (e.g., multiple models); keep output merged.
runs = []
for tb_manifest_path in token_bias_manifest_paths:
    tb_dir = tb_manifest_path.parent
    results_path = tb_dir / "token_bias.results.jsonl"
    if not results_path.exists():
        # Convention fallback: allow a generic results filename for older runs.
        alt = tb_dir / "results.jsonl"
        if alt.exists():
            results_path = alt
        else:
            # NOTE: potential issue: this silently drops an entire run from outputs if the results file is missing.
            print("[WARN] Missing token_bias.results.jsonl next to:", tb_manifest_path)
            continue

    # Prefer a colocated k-sweep manifest if present; otherwise use the embedded block in token_bias.manifest.json.
    ks_path = tb_dir / "k_sweep.manifest.json"
    if not ks_path.exists():
        ks_path = None  # we can still use embedded answer_span_k_sweep from token_bias.manifest.json

    runs.append((tb_manifest_path, results_path, ks_path))

print("Found token score bias runs:", len(runs))
if len(runs) == 0:
    raise RuntimeError("Found token_bias.manifest.json files, but no matching results.jsonl files.")


# ============================================================
# Compute: length normalization AUROC
# and Spearman(score vs answer_len_tokens)
# (Bootstrap removed: point estimates only)
# ============================================================
records = []
k_sweep_records = []

for tb_manifest_path, results_path, ks_path in runs:
    tb_manifest = load_json(tb_manifest_path)
    rows = load_jsonl(results_path)
    if len(rows) == 0:
        # Empty inputs would otherwise produce confusing NaNs/empty plots; skip with explicit warning.
        print("[WARN] Empty results:", results_path)
        continue

    model_tag = tb_manifest_path.parent.name  # Run identifier derived from folder name (used in filenames and grouping).
    model_name = tb_manifest.get("model_name", model_tag)

    # Load arrays (row-order alignment is assumed across all extracted fields).
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    ans_len = np.asarray([float(r["answer_len_tokens"]) for r in rows], dtype=float)

    # Scores expected by this script; update here if the logging schema changes.
    score_cols = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    for c in score_cols:
        if c not in rows[0]:
            raise KeyError(f"Missing required column '{c}' in results: {results_path}")

    # Invariant: each score vector must have length n and match the label/length arrays by row index.
    S_raw = {c: np.asarray([float(r[c]) for r in rows], dtype=float) for c in score_cols}

    n = int(len(y))

    # AUROC + Spearman (direction-corrected)
    for col, s_raw in S_raw.items():
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction  # Apply the same sign convention to length-dependence statistics.

        # Spearman(score vs length) on direction-corrected scores (ρ captures monotonic length bias).
        rho = pd.Series(s).corr(pd.Series(ans_len), method="spearman")

        records.append({
            "run": model_tag,
            "model_name": model_name,
            "n": n,
            "pos_rate": float(y.mean()),
            "score": col,
            "direction": float(direction),

            "auroc": float(au),
            "spearman_score_vs_len": float(rho),

            "token_bias_manifest": str(tb_manifest_path),
            "token_bias_results": str(results_path),
        })

    # k-sweep: prefer explicit k_sweep.manifest.json; otherwise embedded block in token_bias.manifest.json
    if ks_path is not None and ks_path.exists():
        ks_data = load_json(ks_path)
    else:
        ks_data = {"k_list": None, "results": None}
        embedded = tb_manifest.get("answer_span_k_sweep", None)
        if isinstance(embedded, dict):
            ks_data["k_list"] = embedded.get("k_list")
            ks_data["results"] = embedded.get("results")

    # Parse k-sweep (descriptive)
    k_list = ks_data.get("k_list", None)
    ks_results = ks_data.get("results", None)

    if k_list is not None and ks_results is not None:
        for k in k_list:
            kk = str(k)
            if kk not in ks_results:
                # Skip missing k entries to tolerate partial sweeps without failing the full analysis.
                continue
            r = ks_results[kk]
            k_sweep_records.append({
                "run": model_tag,
                "model_name": model_name,
                "k": int(k),
                "n_valid": int(r.get("n_valid", np.nan)),
                "n_skipped": int(r.get("n_skipped", np.nan)),
                "lntp_auc": float(r.get("LNTP_auc", np.nan)),
                "mtp_auc": float(r.get("MTP_auc", np.nan)),
                "k_sweep_manifest": str(ks_path) if ks_path is not None else str(tb_manifest_path),
            })
    else:
        # NOTE: potential issue: missing k-sweep data is not reflected in metrics CSV; only plot/console warns.
        print(f"[WARN] No k-sweep block found for run={model_tag} (neither k_sweep.manifest.json nor embedded).")


df = pd.DataFrame(records)
df_k = pd.DataFrame(k_sweep_records)

csv_metrics = OUT_DIR / "analysis_token_score_bias_metrics.csv"
df.to_csv(csv_metrics, index=False)
print("Wrote:", csv_metrics)

csv_k = OUT_DIR / "analysis_token_score_bias_k_sweep.csv"
df_k.to_csv(csv_k, index=False)
print("Wrote:", csv_k)


# ============================================================
# Plot 1: AUROC mean vs sum (LNTP/MTP) (no CI)
# ============================================================
def plot_auroc_mean_vs_sum(df_run: pd.DataFrame, run_tag: str):
    """Bar plot of AUROC across score variants for a single run (no uncertainty)."""
    order = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    sub = df_run.copy()
    sub["score"] = pd.Categorical(sub["score"], categories=order, ordered=True)
    sub = sub.sort_values("score")  # Ensure stable category order independent of input CSV row order.

    x = np.arange(len(order), dtype=float)
    y = sub["auroc"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 4.7), constrained_layout=True)
    ax.bar(x, y, width=0.65)
    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # AUROC chance baseline.

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Token Score Bias — Length Normalization (AUROC) — {run_tag}")

    # Design choice: dynamic upper limit to keep resolution for high-performing runs while keeping a common floor.
    top = float(np.nanmax(y)) + 0.06  # vorher 0.02
    ax.set_ylim(0.45, max(0.70, top))

    add_value_labels(ax, x, y, fmt="{:.3f}")

    out = OUT_DIR / f"fig_appendix_token_score_bias_auroc_{run_tag}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out)


# ============================================================
# Plot 2: Spearman(score vs length) mean vs sum (no CI)
# ============================================================
def plot_spearman_vs_len(df_run: pd.DataFrame, run_tag: str):
    """Bar plot of Spearman ρ between (direction-corrected) scores and answer length for a single run."""
    order = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    sub = df_run.copy()
    sub["score"] = pd.Categorical(sub["score"], categories=order, ordered=True)
    sub = sub.sort_values("score")  # Stable ordering for across-run comparability.

    x = np.arange(len(order), dtype=float)
    y = sub["spearman_score_vs_len"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 4.7), constrained_layout=True)
    ax.bar(x, y, width=0.65)
    ax.axhline(0.0, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # Zero indicates no monotonic length dependence.

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ(score, answer length)")
    ax.set_title(f"Token Score Bias — Length Dependence (Spearman) — {run_tag}")

    # Symmetric y-limits emphasize direction (+/−) and prevent misleading visual scaling.
    m = float(np.nanmax(np.abs(np.r_[y, [0.0]])))
    ymin = -(m + 0.10)
    ymax =  (m + 0.10)

    # Extra headroom so value labels do not clip at the top edge.
    headroom = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin, ymax + headroom)

    add_value_labels(ax, x, y, fmt="{:.3f}")

    out = OUT_DIR / f"fig_appendix_token_score_bias_spearman_len_{run_tag}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out)


# ============================================================
# Plot 3: k-sweep AUROC vs k (descriptive)
# ============================================================
def plot_k_sweep(dfk_run: pd.DataFrame, run_tag: str):
    """Line plot of k-sweep AUROC (LNTP/MTP) for a single run; restricted to a small, interpretable k set."""
    sub = dfk_run.copy()
    if sub.empty:
        print(f"[WARN] No k-sweep data for run={run_tag}; skipping k-sweep plot.")
        return

    # Restrict k for clarity (avoid overloading the figure); must match the discrete sweep values you want to report.
    K_KEEP = {1, 3, 5, 10, 20}
    sub = sub[sub["k"].isin(K_KEEP)].copy()
    sub = sub.sort_values("k")
    k = sub["k"].to_numpy(dtype=int)
    lntp = sub["lntp_auc"].to_numpy(dtype=float)
    mtp = sub["mtp_auc"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    ax.plot(k, lntp, marker="o", label="LNTP (AUROC)")
    ax.plot(k, mtp, marker="o", label="MTP (AUROC)")
    ax.axhline(0.5, linestyle="--", linewidth=BASELINE_LINEWIDTH)  # AUROC chance baseline.

    ax.set_xlabel("Answer-span k (first k answer tokens)")
    ax.set_ylabel("AUROC (descriptive)")
    ax.set_title(f"Token Score Bias — Answer-span k-sweep (AUROC vs k) — {run_tag}")
    ax.set_xticks(k)

    # Add vertical breathing room so value annotations do not collide with axes/title across backends.
    ax.margins(y=0.12)

    # Annotate per-point values (offset in points -> stable spacing across DPI/export backends).
    for kk, vv in zip(k, lntp):
        ax.annotate(
            f"{vv:.3f}", (kk, vv),
            textcoords="offset points", xytext=(0, 6),
            ha="center", va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE
        )
    for kk, vv in zip(k, mtp):
        ax.annotate(
            f"{vv:.3f}", (kk, vv),
            textcoords="offset points", xytext=(0, -8),
            ha="center", va="top",
            fontsize=VALUE_LABEL_FONTSIZE
        )

    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.12)
    )

    out = OUT_DIR / f"fig_appendix_token_score_bias_k_sweep_{run_tag}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out)


# ============================================================
# Run plots per run_tag (folder)
# ============================================================
if df.empty:
    raise RuntimeError("No metrics computed; check token_bias.results.jsonl parsing.")

for run_tag in sorted(df["run"].unique().tolist()):
    # Invariant: df is grouped by folder-derived run_tag; figures are emitted per run for appendix organization.
    df_run = df[df["run"] == run_tag].copy()
    plot_auroc_mean_vs_sum(df_run, run_tag)
    plot_spearman_vs_len(df_run, run_tag)

    dfk_run = df_k[df_k["run"] == run_tag].copy()
    plot_k_sweep(dfk_run, run_tag)


# ============================================================
# Write a minimal summary CSV (appendix mention)
# ============================================================
def summarize(df: pd.DataFrame):
    """Summarize mean vs sum AUROC per run for quick appendix reporting (including deltas)."""
    # For each run: report AUROC(mean) vs AUROC(sum) and their delta.
    out = []
    for run_tag in sorted(df["run"].unique()):
        # NOTE: potential issue: duplicate rows per (run, score) would make .loc ambiguous and may raise/return a Series.
        # Assumes exactly one row per (run, score); duplicates would make .loc ambiguous.
        sub = df[df["run"] == run_tag].set_index("score")

        def get_val(score):
            # Assumes each score is present exactly once per run; missing keys will raise (preferred over silent NaN).
            r = sub.loc[score]
            return float(r["auroc"])

        l_mean = get_val("lntp_mean")
        l_sum = get_val("lntp_sum")
        m_mean = get_val("mtp_mean")
        m_sum = get_val("mtp_sum")

        out.append({
            "run": run_tag,
            "LNTP_mean_auroc": l_mean,
            "LNTP_sum_auroc": l_sum,
            "LNTP_delta_mean_minus_sum": l_mean - l_sum,

            "MTP_mean_auroc": m_mean,
            "MTP_sum_auroc": m_sum,
            "MTP_delta_mean_minus_sum": m_mean - m_sum,
        })
    return pd.DataFrame(out)

df_summary = summarize(df)
summary_csv = OUT_DIR / "analysis_token_score_bias_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print("Wrote:", summary_csv)

print("[OK] Token score bias ablation done. Outputs in:", OUT_DIR)