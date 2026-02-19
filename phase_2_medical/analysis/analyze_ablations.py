import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final"
ABLATIONS = ROOT / "outputs" / "ablations"

OUT = ROOT / "outputs" / "analysis_ablations"
OUT.mkdir(parents=True, exist_ok=True)

FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

METRICS_OUT = OUT / "metrics"
METRICS_OUT.mkdir(parents=True, exist_ok=True)

# Main scorers reported in Phase 2 (ignore EGH primitives in evaluation)
MAIN_SCORES = {"lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"}

# Map score names from results.jsonl -> manifest["metrics"] keys
SCORE_NAME_MAP = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH_probe_oof",
    "hidden_probe_oof": "Hidden_probe_oof",
}

MODEL_ORDER = ["mistral", "biomistral"]
SCORER_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]


# -----------------------------
# Helpers
# -----------------------------
def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_bootstrap_indices(npz_path: Path):
    z = np.load(npz_path)
    for k in ["indices", "bootstrap_indices", "idx", "I"]:
        if k in z.files:
            return z[k]
    return z[z.files[0]]

def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

def find_label_key(example: dict):
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    # preferred nested
    for k in ["scores", "wb_scores", "whitebox_scores"]:
        if k in example and isinstance(example[k], dict):
            return {kk: float(v) for kk, v in example[k].items() if is_number(v)}

    # fallback: numeric top-level fields with plausible names
    scores = {}
    for k, v in example.items():
        if not is_number(v):
            continue
        kl = k.lower()
        if any(tok in kl for tok in ["lntp", "mtp", "egh", "hidden", "probe", "score", "unc"]):
            scores[k] = float(v)
    return scores

def auroc_with_best_direction(y, s):
    a1 = roc_auc_score(y, s)
    a2 = roc_auc_score(y, -s)
    if a2 > a1:
        return float(a2), -1
    return float(a1), +1

def bootstrap_ci_from_indices(y, s, boot_idx, alpha=0.05):
    B, N = boot_idx.shape
    vals = np.empty(B, dtype=float)
    for b in range(B):
        idx = boot_idx[b]
        vals[b] = roc_auc_score(y[idx], s[idx])
    vals.sort()
    lo = np.quantile(vals, alpha/2)
    hi = np.quantile(vals, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)

def spearman_rho_fn(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman: size mismatch.")

    def _rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1, dtype=float)
        sorted_a = a[order]
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = (i + 1 + j + 1) / 2.0
                ranks[order[i : j + 1]] = avg
            i = j + 1
        return ranks

    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))

def bootstrap_spearman_ci_from_indices(y, s, boot_idx, alpha=0.05):
    B, N = boot_idx.shape
    vals = np.empty(B, dtype=float)
    for b in range(B):
        idx = boot_idx[b]
        vals[b] = spearman_rho_fn(s[idx], y[idx])
    vals.sort()
    lo = np.quantile(vals, alpha/2)
    hi = np.quantile(vals, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)


# -----------------------------
# Run collection
# -----------------------------
def collect_result_files():
    files = []
    files += list(FINAL.glob("*.B*.results.jsonl"))
    files += list(ABLATIONS.glob("**/*.B*.results.jsonl"))
    return sorted(set(files))

def infer_task_model_from_manifest_or_name(manifest, results_path: Path):
    task = str(manifest.get("task", "")).strip().lower()
    model_name = str(manifest.get("config", {}).get("model_name", "")).lower()

    if "biomistral" in model_name:
        model = "biomistral"
    elif "mistral" in model_name:
        model = "mistral"
    else:
        model = "unknown"

    if task not in {"pubmedqa", "medqa"}:
        m = re.match(r"(?P<task>pubmedqa|medqa)_", results_path.name)
        if m:
            task = m["task"]
        else:
            raise ValueError(f"Could not infer task for: {results_path.name}")

    return task, model

def infer_ablation_from_path(results_path: Path, manifest: dict):
    """
    Preferred: manifest["run"]["ablation_name"/"ablation_setting"]
    Fallback: outputs/ablations/<name>/<setting>/... file
    """
    run_info = manifest.get("run", {}) if isinstance(manifest, dict) else {}
    ablation_name = str(run_info.get("ablation_name", "") or "").strip()
    ablation_setting = str(run_info.get("ablation_setting", "") or "").strip()

    # normalize empty -> baseline
    if ablation_name == "":
        ablation_name = "baseline"
    if ablation_setting == "":
        ablation_setting = "baseline"

    # if it says baseline but file is inside outputs/ablations, infer from folder structure
    p = str(results_path).replace("\\", "/")
    if ablation_name == "baseline" and "/outputs/ablations/" in p:
        parts = p.split("/")
        try:
            i = parts.index("ablations")
            ablation_name = parts[i + 1]
            ablation_setting = parts[i + 2]
        except Exception:
            pass

    return ablation_name, ablation_setting


# -----------------------------
# Plot helpers
# -----------------------------
def _add_bar_labels(ax, bars, values, y_min=None, y_max=None, fmt="{:.3f}"):
    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()
    rng = max(1e-9, (y_max - y_min))
    pad = 0.03 * rng

    for b, v in zip(bars, values):
        if v is None or not np.isfinite(v):
            continue
        x = b.get_x() + b.get_width() / 2.0
        if v >= 0:
            y = v - pad
            va = "top"
        else:
            y = v + pad
            va = "bottom"
        ax.text(x, y, fmt.format(v), ha="center", va=va, fontsize=9, color="white")

def plot_grouped_bars(df_task: pd.DataFrame, metric_col: str, lo_col: str, hi_col: str,
                      title: str, outpath: Path, ylims, hline=None):
    dfp = df_task.copy()
    dfp["score"] = dfp["score"].str.lower()
    dfp["model"] = dfp["model"].str.lower()

    scorers = [s for s in SCORER_ORDER if s in set(dfp["score"])]
    models = [m for m in MODEL_ORDER if m in set(dfp["model"])]
    if len(scorers) == 0 or len(models) == 0:
        print(f"[WARN] No data for plot: {title}")
        return

    def get_row(score, model):
        sub = dfp[(dfp["score"] == score) & (dfp["model"] == model)]
        if len(sub) == 0:
            return None
        # if duplicates exist (e.g. multiple ablations), expect caller filtered already
        return sub.iloc[0]

    x = np.arange(len(scorers), dtype=float)
    width = 0.38 if len(models) == 2 else 0.6

    plt.figure(figsize=(10, 4.8))
    ax = plt.gca()

    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2.0) * width
        xs = x + offset

        vals, yerr_low, yerr_high = [], [], []
        for sc in scorers:
            r = get_row(sc, model)
            if r is None:
                vals.append(np.nan); yerr_low.append(0.0); yerr_high.append(0.0)
            else:
                v = float(r[metric_col])
                lo = float(r[lo_col]); hi = float(r[hi_col])
                vals.append(v)
                yerr_low.append(max(0.0, v - lo))
                yerr_high.append(max(0.0, hi - v))

        bars = ax.bar(xs, vals, width=width, label=model.upper())
        ax.errorbar(xs, vals, yerr=[yerr_low, yerr_high], fmt="none", capsize=3)
        _add_bar_labels(ax, bars, vals, y_min=ylims[0], y_max=ylims[1], fmt="{:.3f}")

    if hline is not None:
        ax.axhline(hline, linewidth=1, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(scorers, rotation=0)
    ax.set_ylim(*ylims)
    ax.set_title(title)
    ax.legend(title="Model", frameon=False)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print("Saved:", outpath)


# -----------------------------
# Main
# -----------------------------
def main():
    runs = []
    for results_path in collect_result_files():
        stem = results_path.name.replace(".results.jsonl", "")
        manifest_path = results_path.with_name(f"{stem}.manifest.json")
        boot_path = results_path.with_name(f"{stem}.manifest.bootstrap_indices.npz")

        if not manifest_path.exists():
            print(f"[SKIP] Missing manifest: {manifest_path}")
            continue
        if not boot_path.exists():
            print(f"[SKIP] Missing bootstrap npz: {boot_path}")
            continue

        manifest = load_manifest(manifest_path)
        task, model = infer_task_model_from_manifest_or_name(manifest, results_path)
        ablation_name, ablation_setting = infer_ablation_from_path(results_path, manifest)

        runs.append((task, model, ablation_name, ablation_setting, results_path, manifest_path, boot_path))

    if not runs:
        raise RuntimeError("No runnable result files found (need results.jsonl + manifest.json + bootstrap npz).")

    print(f"Found {len(runs)} runs.")
    print("Example:", runs[0][:4], runs[0][4].name)

    # ---- Compute metrics ----
    auroc_rows = []
    spear_rows = []

    for task, model, ablation_name, ablation_setting, results_path, manifest_path, boot_path in runs:
        rows = load_jsonl(results_path)
        if not rows:
            print(f"[SKIP] Empty results: {results_path}")
            continue

        manifest = load_manifest(manifest_path)
        y_key = find_label_key(rows[0])
        y = np.array([int(r[y_key]) for r in rows], dtype=int)

        score_dicts = [extract_scores(r) for r in rows]
        keys = set(score_dicts[0].keys())
        for d in score_dicts[1:]:
            keys &= set(d.keys())
        keys = sorted(keys)

        if not keys:
            print(f"[SKIP] No scores detected in: {results_path}")
            continue

        S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
        # keep only main scores (if present)
        S = {k.lower(): v for k, v in S.items()}
        S = {k: v for k, v in S.items() if k in MAIN_SCORES}

        boot_idx = load_bootstrap_indices(boot_path)

        for score_name, s_raw in S.items():
            au, direction = auroc_with_best_direction(y, s_raw)
            s_oriented = s_raw * direction

            mean_b, lo, hi = bootstrap_ci_from_indices(y, s_oriented, boot_idx, alpha=0.05)
            auroc_rows.append({
                "task": task,
                "model": model,
                "ablation_name": ablation_name,
                "ablation_setting": ablation_setting,
                "score": score_name,
                "direction": direction,
                "auroc": au,
                "boot_mean": mean_b,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "N": int(len(y)),
                "pos_rate": float(y.mean()),
                "results_file": str(results_path),
            })

            # Spearman from manifest (optional)
            metric_key = SCORE_NAME_MAP.get(score_name)
            if metric_key is not None:
                rho_manifest = float(manifest.get("metrics", {}).get(metric_key, {}).get("spearman_rho", np.nan))
            else:
                rho_manifest = np.nan

            # Spearman bootstrap CI computed from results (preferred)
            rho_mean, rho_lo, rho_hi = bootstrap_spearman_ci_from_indices(y, s_oriented, boot_idx, alpha=0.05)

            spear_rows.append({
                "task": task,
                "model": model,
                "ablation_name": ablation_name,
                "ablation_setting": ablation_setting,
                "score": score_name,
                "spearman_rho_manifest": rho_manifest,
                "spearman_rho_boot_mean": rho_mean,
                "ci95_lo": rho_lo,
                "ci95_hi": rho_hi,
                "N": int(len(y)),
                "pos_rate": float(y.mean()),
                "results_file": str(results_path),
            })

    df_auroc = pd.DataFrame(auroc_rows)
    df_spear = pd.DataFrame(spear_rows)

    # ---- write CSVs ----
    out1 = METRICS_OUT / "ablations_auroc_ci.csv"
    out2 = METRICS_OUT / "ablations_spearman_ci.csv"
    df_auroc.to_csv(out1, index=False)
    df_spear.to_csv(out2, index=False)
    print("Wrote:", out1)
    print("Wrote:", out2)

    # ---- sanity summary ----
    # helpful quick check: missing scorers per run
    print("\n[Sanity] Missing score checks:")
    for (task, model, ab_name, ab_set, results_path, _, _) in runs:
        sub = df_auroc[(df_auroc.task == task) & (df_auroc.model == model) &
                      (df_auroc.ablation_name == ab_name) & (df_auroc.ablation_setting == ab_set)]
        present = set(sub["score"].tolist())
        missing = sorted(MAIN_SCORES - present)
        if missing:
            print(f" - {task}/{model} {ab_name}/{ab_set}: missing {missing}")

    # ---- Plots per ablation_name (and per task) ----
    for ab_name in sorted(df_auroc["ablation_name"].unique()):
        dfA_auc = df_auroc[df_auroc["ablation_name"] == ab_name].copy()
        dfA_sp = df_spear[df_spear["ablation_name"] == ab_name].copy()

        for task in sorted(dfA_auc["task"].unique()):
            # For grouped plot we need one row per (task, model, score) => pick best setting or enforce filter.
            # Here: pick the BEST AUROC setting per (task,model,score) so you see "does ablation have signal".
            df_task_auc = (
                dfA_auc[dfA_auc["task"] == task]
                .sort_values("auroc", ascending=False)
                .groupby(["task", "model", "score"], as_index=False)
                .head(1)
            )
            df_task_sp = (
                dfA_sp[dfA_sp["task"] == task]
                .sort_values("spearman_rho_boot_mean", ascending=False)
                .groupby(["task", "model", "score"], as_index=False)
                .head(1)
            )

            out_auc = FIGS / f"ablation_{ab_name}_AUROC_grouped_{task}.pdf"
            plot_grouped_bars(
                df_task_auc,
                metric_col="auroc",
                lo_col="ci95_lo",
                hi_col="ci95_hi",
                title=f"{ab_name} — AUROC + 95% CI (best setting per score) — {task}",
                outpath=out_auc,
                ylims=(0.0, 1.0),
                hline=0.5
            )

            out_sp = FIGS / f"ablation_{ab_name}_Spearman_grouped_{task}.pdf"
            plot_grouped_bars(
                df_task_sp,
                metric_col="spearman_rho_boot_mean",
                lo_col="ci95_lo",
                hi_col="ci95_hi",
                title=f"{ab_name} — Spearman ρ + 95% CI (best setting per score) — {task}",
                outpath=out_sp,
                ylims=(-1.0, 1.0),
                hline=0.0
            )

        # Also write "best setting per score" tables for each ablation
        best_auc = (
            dfA_auc.sort_values("auroc", ascending=False)
            .groupby(["task","model","score"], as_index=False)
            .head(1)
            .sort_values(["task","model","score"])
        )
        best_sp = (
            dfA_sp.sort_values("spearman_rho_boot_mean", ascending=False)
            .groupby(["task","model","score"], as_index=False)
            .head(1)
            .sort_values(["task","model","score"])
        )
        best_auc.to_csv(METRICS_OUT / f"best_settings_{ab_name}_auroc.csv", index=False)
        best_sp.to_csv(METRICS_OUT / f"best_settings_{ab_name}_spearman.csv", index=False)

    print("\nDone. Outputs in:", OUT)

if __name__ == "__main__":
    main()