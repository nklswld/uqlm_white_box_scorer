# phase_2_medical/analysis/ablations/analyze_token_score_bias.py
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
FONT_SCALE = 1.35  # keep consistent with your other scripts

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
# Helper: labels above bars (no CI)
# ============================================================
def add_value_labels(ax, x_positions, y_values, fmt="{:.3f}", fontsize=None):
    if fontsize is None:
        fontsize = VALUE_LABEL_FONTSIZE

    for x, y in zip(x_positions, y_values):
        if y is None or (isinstance(y, float) and np.isnan(y)):
            continue
        ax.annotate(
            fmt.format(float(y)),
            (float(x), float(y)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# ============================================================
# IO helpers
# ============================================================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: Path):
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
    au = roc_auc_score(y, s)
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0


# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../phase_2_medical
ABL_ROOT = ROOT / "outputs" / "ablations" / "token_score_bias"
OUT_DIR = ROOT / "outputs" / "figs" / "ablations" / "token_score_bias"
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

if len(token_bias_manifest_paths) == 0:
    listing = sorted([str(p.relative_to(ABL_ROOT)) for p in ABL_ROOT.glob("**/*")][:250])
    raise FileNotFoundError("No token_bias.manifest.json found. Sample listing:\n" + "\n".join(listing))

# If multiple are present, analyze all (e.g., multiple models); keep output merged.
runs = []
for tb_manifest_path in token_bias_manifest_paths:
    tb_dir = tb_manifest_path.parent
    results_path = tb_dir / "token_bias.results.jsonl"
    if not results_path.exists():
        # fallback if user named it results.jsonl
        alt = tb_dir / "results.jsonl"
        if alt.exists():
            results_path = alt
        else:
            print("[WARN] Missing token_bias.results.jsonl next to:", tb_manifest_path)
            continue

    # try to find k_sweep.manifest.json in same dir
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
        print("[WARN] Empty results:", results_path)
        continue

    # Identify model "tag" for filenames (folder name is usually 'mistral', etc.)
    model_tag = tb_manifest_path.parent.name  # e.g. mistral/
    model_name = tb_manifest.get("model_name", model_tag)

    # Load arrays
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    ans_len = np.asarray([float(r["answer_len_tokens"]) for r in rows], dtype=float)

    # Scores present in your example
    score_cols = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    for c in score_cols:
        if c not in rows[0]:
            raise KeyError(f"Missing required column '{c}' in results: {results_path}")

    S_raw = {c: np.asarray([float(r[c]) for r in rows], dtype=float) for c in score_cols}

    n = int(len(y))

    # AUROC + Spearman (direction-corrected)
    for col, s_raw in S_raw.items():
        au, direction = auroc_with_best_direction(y, s_raw)
        s = s_raw * direction

        # Spearman(score vs length) on direction-corrected scores
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
    order = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    sub = df_run.copy()
    sub["score"] = pd.Categorical(sub["score"], categories=order, ordered=True)
    sub = sub.sort_values("score")

    x = np.arange(len(order), dtype=float)
    y = sub["auroc"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 4.7), constrained_layout=True)
    ax.bar(x, y)
    ax.axhline(0.5, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Token Score Bias — Length Normalization (AUROC) — {run_tag}")

    # keep your dynamic ylim behavior
    ax.set_ylim(0.45, max(0.65, float(np.nanmax(y) + 0.02)))

    add_value_labels(ax, x, y, fmt="{:.3f}")

    out = OUT_DIR / f"fig_appendix_token_score_bias_auroc_{run_tag}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out)


# ============================================================
# Plot 2: Spearman(score vs length) mean vs sum (no CI)
# ============================================================
def plot_spearman_vs_len(df_run: pd.DataFrame, run_tag: str):
    order = ["lntp_mean", "lntp_sum", "mtp_mean", "mtp_sum"]
    sub = df_run.copy()
    sub["score"] = pd.Categorical(sub["score"], categories=order, ordered=True)
    sub = sub.sort_values("score")

    x = np.arange(len(order), dtype=float)
    y = sub["spearman_score_vs_len"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11.0, 4.7), constrained_layout=True)
    ax.bar(x, y)
    ax.axhline(0.0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ(score, answer length)")
    ax.set_title(f"Token Score Bias — Length Dependence (Spearman) — {run_tag}")

    # keep your symmetric ylim behavior
    m = float(np.nanmax(np.abs(np.r_[y, [0.0]])))
    ax.set_ylim(-(m + 0.05), (m + 0.05))

    add_value_labels(ax, x, y, fmt="{:.3f}")

    out = OUT_DIR / f"fig_appendix_token_score_bias_spearman_len_{run_tag}.pdf"
    safe_savefig(fig, out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote:", out)


# ============================================================
# Plot 3: k-sweep AUROC vs k (descriptive)
# ============================================================
def plot_k_sweep(dfk_run: pd.DataFrame, run_tag: str):
    sub = dfk_run.copy()
    if sub.empty:
        print(f"[WARN] No k-sweep data for run={run_tag}; skipping k-sweep plot.")
        return

    # Restrict k for clarity (avoid overloading the figure)
    K_KEEP = {1, 3, 5, 10, 20}
    sub = sub[sub["k"].isin(K_KEEP)].copy()
    sub = sub.sort_values("k")
    k = sub["k"].to_numpy(dtype=int)
    lntp = sub["lntp_auc"].to_numpy(dtype=float)
    mtp = sub["mtp_auc"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    ax.plot(k, lntp, marker="o", label="LNTP (AUROC)")
    ax.plot(k, mtp, marker="o", label="MTP (AUROC)")
    ax.axhline(0.5, linestyle="--", linewidth=1)

    ax.set_xlabel("Answer-span k (first k answer tokens)")
    ax.set_ylabel("AUROC (descriptive)")
    ax.set_title(f"Token Score Bias — Answer-span k-sweep (AUROC vs k) — {run_tag}")
    ax.set_xticks(k)

    # Give vertical breathing room so text doesn't collide with axes/title
    ax.margins(y=0.12)

    # annotate values (offset in points -> stable spacing)
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

    ax.legend(frameon=False)

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
    df_run = df[df["run"] == run_tag].copy()
    plot_auroc_mean_vs_sum(df_run, run_tag)
    plot_spearman_vs_len(df_run, run_tag)

    dfk_run = df_k[df_k["run"] == run_tag].copy()
    plot_k_sweep(dfk_run, run_tag)


# ============================================================
# Write a minimal summary CSV (appendix mention)
# ============================================================
def summarize(df: pd.DataFrame):
    # For each run: report AUROC(mean) vs AUROC(sum) and their delta
    out = []
    for run_tag in sorted(df["run"].unique()):
        sub = df[df["run"] == run_tag].set_index("score")

        def get_val(score):
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