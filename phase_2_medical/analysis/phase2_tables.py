"""Phase 2 table generation and metric summarization for medical error detection.

This module (i) aggregates per-run bootstrap summaries (AUROC and Spearman rho) and
(ii) renders publication-ready wide CSV tables for per-task and appendix views.
Inputs: per-run artifacts in outputs/final (*.manifest.json, *.results.jsonl, *.npz) and/or
precomputed metric CSVs in outputs/figures_tables/tables_general.
Outputs: wide-format table CSVs in outputs/figures_tables/tables_general (+ appendix/).
Determinism: fully deterministic given fixed input files; bootstrap resamples are read from disk.
"""

# phase_2_medical/analysis/phase2_tables.py
import pandas as pd
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------
# Paths
# ----------------------------
# Base directory for upstream Phase 2 per-run artifacts (produced elsewhere).
BASE = (Path(__file__).resolve().parents[1] / "outputs" / "final")
# Output directory for generated metric CSVs and wide tables.
OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs" / "figures_tables" / "tables_general")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Appendix outputs: "combined" tables spanning tasks for a fixed scorer.
APPENDIX_DIR = OUT_DIR / "appendix"
APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

# Metric summary inputs (must contain task/model/score keys plus metric + CI columns).
AUROC_CSV = OUT_DIR / "phase2_metrics_auroc_ci.csv"
SPEAR_CSV = OUT_DIR / "phase2_metrics_spearman_rho.csv"


# ----------------------------
# Config
# ----------------------------
# Canonical display order for tables; non-listed items may still appear if present in input.
TASK_ORDER = ["medqa", "pubmedqa"]
MODEL_ORDER = ["mistral", "biomistral"]
SCORER_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]

# Stable, repository-wide labels for table readability.
SCORER_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}

MODEL_PRETTY = {
    "mistral": "Mistral",
    "biomistral": "BioMistral",
}

# Task labels used in combined appendix tables.
TASK_PRETTY = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
}

DIGITS = 2


# ----------------------------
# Helpers
# ----------------------------
def mean_ci_str(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    """Format as 'mean ± halfwidth'; returns '—' if any input is NaN."""
    if any(pd.isna(x) for x in [mean, lo, hi]):
        return "—"
    # CI input is [lo, hi]; manuscript convention reports half-width (hi-lo)/2.
    pm = (hi - lo) / 2.0
    return f"{mean:.{digits}f} ± {pm:.{digits}f}"

def _get_row(
    df: pd.DataFrame,
    task: str,
    model: str,
    score: str,
    prefer_col: str | None = None,
) -> pd.Series | None:
    """Select the unique row for (task, model, score); break ties by prefer_col if provided."""
    r = df[(df["task"] == task) & (df["model"] == model) & (df["score"] == score)]
    if len(r) == 0:
        return None

    if len(r) > 1 and prefer_col is not None:
        # NOTE: potential issue: duplicates indicate upstream aggregation problems; we pick deterministically.
        print(f"[WARN] Duplicate rows for ({task},{model},{score}) -> taking max {prefer_col}")

    # If duplicates exist, pick deterministically (best by prefer_col).
    if prefer_col is not None and prefer_col in r.columns and len(r) > 1:
        r = r.sort_values(prefer_col, ascending=False)

    return r.iloc[0]

def build_wide_table(
    df: pd.DataFrame,
    value_cols: tuple[str, str, str],  # (mean_col, lo_col, hi_col)
    task: str,
    digits: int = 3,
) -> pd.DataFrame:
    """Render a per-task table: scorers as rows, models as columns, cells formatted via mean_ci_str()."""
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for score in SCORER_ORDER:
        row = {"Scorer": SCORER_PRETTY.get(score, score)}
        for model in MODEL_ORDER:
            # Alignment invariant: each cell corresponds to exactly one (task, model, score) triple.
            r = _get_row(df, task, model, score, prefer_col=mean_col)
            if r is None:
                # Explicit placeholder keeps table shape stable across runs/tasks.
                row[MODEL_PRETTY[model]] = "—"
            else:
                row[MODEL_PRETTY[model]] = mean_ci_str(
                    float(r[mean_col]),
                    float(r[lo_col]),
                    float(r[hi_col]),
                    digits=digits,
                )
        rows.append(row)

    return pd.DataFrame(rows)



# Optional out_dir so combined tables can be routed to the appendix subdirectory.
def save_table_bundle(wide: pd.DataFrame, name: str, caption: str, label: str, out_dir: Path = OUT_DIR):
    """Persist a single wide table as CSV; LaTeX wrapping/metadata is handled elsewhere."""
    csv_path = out_dir / f"{name}.csv"
    wide.to_csv(csv_path, index=False)

    print(f"[OK] Wrote: {csv_path}")


# Combined ALL-task table per scorer (appendix convenience view).
def build_all_task_table_per_scorer(
    df: pd.DataFrame,
    value_cols: tuple[str, str, str],  # (mean_col, lo_col, hi_col)
    score: str,
    digits: int = 3,
) -> pd.DataFrame:
    """Render an appendix table: tasks as rows, models as columns, for a fixed scorer."""
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for task in TASK_ORDER:
        row = {"Task": TASK_PRETTY.get(task, task)}
        for model in MODEL_ORDER:
            # Alignment invariant: each cell corresponds to exactly one (task, model, score) triple.
            r = _get_row(df, task, model, score, prefer_col=mean_col)
            if r is None:
                row[MODEL_PRETTY[model]] = "—"
            else:
                row[MODEL_PRETTY[model]] = mean_ci_str(
                    float(r[mean_col]),
                    float(r[lo_col]),
                    float(r[hi_col]),
                    digits=digits,
                )
        rows.append(row)

    return pd.DataFrame(rows)



# ----------------------------
# Metric CSV generation (moved from phase2_figures.py)
# ----------------------------
# Scores explicitly supported for Phase 2 reporting; other numeric fields are ignored.
MAIN_SCORES = {"lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"}

def load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts; blank lines are skipped."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_bootstrap_map(npz_path: Path):
    """Load bootstrap resample indices from .npz, normalizing common key aliases to 'indices'."""
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in z.files:
        arr = z[k]
        # Some writers store a list/array of arrays (dtype=object); stack to (B, N).
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = np.stack(arr, axis=0)
        if isinstance(arr, np.ndarray):
            out[str(k).lower()] = arr.astype(int)

    # Accept multiple naming conventions for the main bootstrap index matrix.
    for alias in ["indices", "boot_idx", "bootstrap_indices", "idx"]:
        if alias in out and "indices" not in out:
            out["indices"] = out[alias]

    # Heuristic: if a single array exists, treat it as the indices matrix.
    if "indices" not in out and len(out) == 1:
        out["indices"] = next(iter(out.values()))

    return out

def find_label_key(example: dict):
    """Infer the binary label field name from a single example row."""
    for k in ["is_error", "label", "y", "target", "error"]:
        if k in example:
            return k
    # Fail fast: downstream metrics require a binary label vector.
    raise KeyError("No label key found (expected is_error/label/y/target/error).")

def extract_scores(example: dict):
    """Extract per-scorer numeric scores from a row, supporting both nested and flat conventions."""
    if "scores" in example and isinstance(example["scores"], dict):
        return example["scores"]
    if "wb_scores" in example and isinstance(example["wb_scores"], dict):
        return example["wb_scores"]

    # Fallback: scan numeric fields and keep those whose key contains a known scorer token.
    scores = {}
    for k, v in example.items():
        if isinstance(v, (float, int)):
            kk = str(k).lower()
            if any(s in kk for s in ["lntp", "mtp", "egh", "hidden"]):
                scores[kk] = float(v)
    return scores

def auroc_with_best_direction(y: np.ndarray, s: np.ndarray):
    """Compute AUROC and flip score sign if needed so AUROC >= 0.5; returns (auroc, direction)."""
    # AUROC undefined if only one class is present
    if np.min(y) == np.max(y):
        return np.nan, +1.0

    au = roc_auc_score(y, s)
    # Convention: direction indicates the sign applied to raw scores before reporting/bootstrapping.
    if au < 0.5:
        return roc_auc_score(y, -s), -1.0
    return au, +1.0

def bootstrap_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and two-sided (1-alpha) quantile CI from explicit resample indices."""
    aucs = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Skip degenerate resamples (single-class); affects effective bootstrap size.
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, ss))

    aucs = np.asarray(aucs, dtype=float)
    if aucs.size == 0:
        # All resamples degenerate -> no meaningful CI; caller renders as missing.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(aucs))
    lo = float(np.quantile(aucs, alpha / 2))
    hi = float(np.quantile(aucs, 1 - alpha / 2))
    return mean, lo, hi

def bootstrap_spearman_ci_from_indices(y: np.ndarray, s: np.ndarray, boot_idx: np.ndarray, alpha=0.05):
    """Bootstrap mean and two-sided (1-alpha) quantile CI for Spearman rho from resample indices."""
    rhos = []
    for idx in boot_idx:
        yy = y[idx]
        ss = s[idx]
        # Skip degenerate resamples (single-class); Spearman may be undefined/unstable.
        if yy.min() == yy.max():
            continue
        rho = pd.Series(ss).corr(pd.Series(yy), method="spearman")
        if pd.isna(rho):
            continue
        rhos.append(float(rho))

    rhos = np.asarray(rhos, dtype=float)
    if rhos.size == 0:
        # All resamples degenerate/NaN -> no meaningful CI; caller renders as missing.
        return np.nan, np.nan, np.nan

    mean = float(np.mean(rhos))
    lo = float(np.quantile(rhos, alpha / 2))
    hi = float(np.quantile(rhos, 1 - alpha / 2))
    return mean, lo, hi

def infer_task_model_from_manifest(manifest_path: Path):
    """Infer task and coarse model family from the run manifest JSON."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    task = str(m.get("task", "")).lower()
    config = m.get("config", {})
    model_name = str(config.get("model_name", "")).lower()
    # Heuristic: any model_name containing "bio" maps to the BioMistral family label.
    model = "biomistral" if "bio" in model_name else "mistral"
    return task, model

def generate_phase2_metric_csvs(final_dir: Path, out_dir: Path):
    """
    Generate Phase 2 metric summaries from per-run artifacts and write:
      - phase2_metrics_auroc_ci.csv
      - phase2_metrics_spearman_rho.csv
    """
    final_dir = Path(final_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not final_dir.exists():
        raise FileNotFoundError(f"FINAL directory not found: {final_dir}")

    runs = []
    for manifest_path in sorted(final_dir.glob("*.manifest.json")):
        results_path = manifest_path.with_suffix("").with_suffix(".results.jsonl")
        boot_path = manifest_path.with_suffix("").with_suffix(".manifest.bootstrap_indices.npz")

        if not results_path.exists():
            print("[WARN] Missing results for", manifest_path.name, "expected:", results_path.name)
            continue
        if not boot_path.exists():
            print("[WARN] Missing bootstrap npz for", manifest_path.name, "expected:", boot_path.name)
            continue

        task, model = infer_task_model_from_manifest(manifest_path)
        runs.append((task, model, results_path, manifest_path, boot_path))

    if len(runs) == 0:
        listing = sorted([p.name for p in final_dir.iterdir()])
        raise RuntimeError(
            "Found runs: [] but FINAL contains files. Naming mismatch?\n"
            f"FINAL listing:\n{listing}\n"
            "Expected per run:\n"
            "  *.manifest.json\n"
            "  sameprefix.results.jsonl\n"
            "  sameprefix.manifest.bootstrap_indices.npz\n"
        )

    records = []
    spearman_records = []
    spearman_ci_rows = []

    SCORE_TO_BOOTKEY = {
        "lntp": "lntp",
        "mtp": "mtp",
        "egh_probe_oof": "egh",
        "hidden_probe_oof": "hidden",
    }

    for task, model, results_path, manifest_path, boot_path in runs:
        rows = load_jsonl(results_path)
        if not rows:
            # No examples -> no metrics; skip run without failing the entire batch.
            continue

        # Label extraction convention is inferred from the first row and applied uniformly.
        y_key = find_label_key(rows[0])
        y = np.array([int(r[y_key]) for r in rows], dtype=int)

        # Score extraction supports nested dicts and flat numeric fields; keys normalized to lowercase.
        score_dicts = [{str(k).lower(): v for k, v in extract_scores(r).items()} for r in rows]

        # Intersection enforces per-example completeness: only scorers present in all rows are kept.
        keys = set(score_dicts[0].keys())
        for d in score_dicts[1:]:
            keys &= set(d.keys())
        keys = sorted(keys)

        S = {k: np.array([d[k] for d in score_dicts], dtype=float) for k in keys}
        # Only report the configured main scores to keep tables comparable across runs.
        S = {k: v for k, v in S.items() if k in MAIN_SCORES}

        boot_map = load_bootstrap_map(boot_path)
        hidden_kept = boot_map.get("hidden_kept_indices", None)

        for score_name, s_raw in S.items():
            score_l = str(score_name).lower()
            boot_key = SCORE_TO_BOOTKEY.get(score_l, score_l)
            boot_idx = boot_map.get(boot_key, None)
            if boot_idx is None:
                # Fallback: use shared bootstrap indices if scorer-specific indices are absent.
                boot_idx = boot_map.get("indices", None)
                
            if boot_idx is None:
                # Without explicit resamples, results are non-reproducible; skip rather than improvise.
                print(f"[WARN] No bootstrap indices for score={score_l} in {boot_path.name}; skipping.")
                continue

            # Determine sign convention from AUROC on the relevant subset, then apply consistently.
            if score_l == "hidden_probe_oof" and hidden_kept is not None:
                au, direction = auroc_with_best_direction(y[hidden_kept], s_raw[hidden_kept])
            else:
                au, direction = auroc_with_best_direction(y, s_raw)

            if np.isnan(au):
                # Single-class label vector makes AUROC undefined; skip to avoid misleading summaries.
                print(f"[WARN] AUROC undefined (single-class y) for task={task}, model={model}, score={score_l}. Skipping.")
                continue

            # Apply direction so that higher scores correspond to the "positive" label for reporting.
            s = s_raw * direction

            if score_l == "hidden_probe_oof":
                # Hidden probe may include NaNs or a curated subset; both affect N and bootstrap alignment.
                if hidden_kept is None:
                    m = np.isfinite(s)
                    y_use = y[m]
                    s_use = s[m]
                else:
                    y_use = y[hidden_kept]
                    s_use = s[hidden_kept]
            else:
                y_use = y
                s_use = s

            # Guardrail: bootstrap index length must match the exact vector used for metrics.
            if boot_idx.shape[1] != len(y_use):
                raise ValueError(
                    f"Bootstrap shape mismatch for {score_l}: boot_idx {boot_idx.shape} vs N={len(y_use)} "
                    f"(boot_file={boot_path.name}, boot_key={boot_key})"
                )

            mean_b, lo, hi = bootstrap_ci_from_indices(y_use, s_use, boot_idx, alpha=0.05)

            # Spearman rho is computed against the binary label (rank correlation with error indicator).
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
                "manifest_file": manifest_path.relative_to(REPO_ROOT).as_posix(),
                "results_file": results_path.relative_to(REPO_ROOT).as_posix(),
                "boot_file": boot_path.relative_to(REPO_ROOT).as_posix(),
            })

    if len(records) == 0:
        raise RuntimeError(
            "No AUROC records produced. All runs may have been skipped "
            "(missing results/bootstrap files, missing indices, or all scorers filtered out)."
        )
        
    df_au = pd.DataFrame(records).sort_values(["task", "model", "auroc"], ascending=[True, True, False])
    out_au = out_dir / "phase2_metrics_auroc_ci.csv"
    df_au.to_csv(out_au, index=False)
    print("[OK] Wrote:", out_au)

    df_spear_main = pd.DataFrame(spearman_records)
    df_spear_ci = pd.DataFrame(spearman_ci_rows)
    # Left-join preserves per-run rows even if CI computation returned NaNs.
    df_spear_main = df_spear_main.merge(df_spear_ci, on=["task", "model", "score"], how="left")

    out_sp = out_dir / "phase2_metrics_spearman_rho.csv"
    df_spear_main.to_csv(out_sp, index=False)
    print("[OK] Wrote:", out_sp)



# ----------------------------
# Main
# ----------------------------
def main():
    """Generate metric CSVs (if missing) and emit per-task + appendix wide tables."""
    
    # Primary control: prefer existing metric CSVs; regenerate only when absent.
    if not AUROC_CSV.exists() or not SPEAR_CSV.exists():
        generate_phase2_metric_csvs(final_dir=BASE, out_dir=OUT_DIR)
    else:
        print("[OK] Metric CSVs exist -> skipping regeneration.")
    # --- AUROC ---
    if not AUROC_CSV.exists():
        raise FileNotFoundError(f"Missing AUROC CSV: {AUROC_CSV}")
    df_au = pd.read_csv(AUROC_CSV)
    
    # Canonicalize join keys to lowercase for stable matching against configured orders.
    for k in ["task", "model", "score"]:
        df_au[k] = df_au[k].astype(str).str.lower()
    
    # --- column aliasing (phase2_figures.py writes 'auroc_boot_mean', tables expect 'boot_mean')
    if "boot_mean" not in df_au.columns:
        if "auroc_boot_mean" in df_au.columns:
            df_au = df_au.rename(columns={"auroc_boot_mean": "boot_mean"})
        elif "auroc" in df_au.columns:
            # NOTE: potential issue: using point AUROC as bootstrap mean changes interpretation of '±' cells.
            # fallback: use point estimate as "boot_mean" if bootstrap mean is not available
            df_au["boot_mean"] = df_au["auroc"]

    # Schema guardrails: fail fast to avoid silently producing malformed tables.
    for c in ["task", "model", "score", "boot_mean", "ci95_lo", "ci95_hi"]:
        if c not in df_au.columns:
            raise KeyError(f"AUROC CSV missing column '{c}'. Found: {list(df_au.columns)}")

    # --- Spearman ---
    if not SPEAR_CSV.exists():
        raise FileNotFoundError(f"Missing Spearman CSV: {SPEAR_CSV}")
    df_sp = pd.read_csv(SPEAR_CSV)
    # Canonicalize join keys to lowercase for stable matching against configured orders.
    for k in ["task", "model", "score"]:
        df_sp[k] = df_sp[k].astype(str).str.lower()

    # Schema guardrails: metric column name differs from AUROC; CI columns are shared.
    for c in ["task", "model", "score", "spearman_rho_boot_mean", "ci95_lo", "ci95_hi"]:
        if c not in df_sp.columns:
            raise KeyError(f"Spearman CSV missing column '{c}'. Found: {list(df_sp.columns)}")

    # Build per-task tables, preferring TASK_ORDER but falling back to what's present in the inputs.
    tasks = [t for t in TASK_ORDER if t in set(df_au["task"].unique()) or t in set(df_sp["task"].unique())]
    if not tasks:
        # NOTE: potential issue: fallback ordering is alphabetical, which may differ from manuscript ordering.
        tasks = sorted(set(df_au["task"].unique()) | set(df_sp["task"].unique()))

    for task in tasks:
        # AUROC table
        wide_au = build_wide_table(
            df_au,
            value_cols=("boot_mean", "ci95_lo", "ci95_hi"),
            task=task,
            digits=DIGITS,
        )
        save_table_bundle(
            wide_au,
            name=f"tab_phase2_auroc_{task}",
            caption=f"Phase 2 AUROC (mean ± 95\\% CI half-width) for error detection on {task.upper()}.",
            label=f"tab:phase2_auroc_{task}",
        )

        # Spearman table
        wide_sp = build_wide_table(
            df_sp,
            value_cols=("spearman_rho_boot_mean", "ci95_lo", "ci95_hi"),
            task=task,
            digits=DIGITS,
        )
        save_table_bundle(
            wide_sp,
            name=f"tab_phase2_spearman_{task}",
            caption=f"Phase 2 Spearman $\\rho$ (mean ± 95\\% CI half-width) between score and error label on {task.upper()}.",
            label=f"tab:phase2_spearman_{task}",
        )

    # Combined tables -> appendix (one table per scorer, with tasks as rows).
    for score in SCORER_ORDER:
        # AUROC combined
        wide_au_all = build_all_task_table_per_scorer(
            df_au,
            value_cols=("boot_mean", "ci95_lo", "ci95_hi"),
            score=score,
            digits=DIGITS,
        )
        save_table_bundle(
            wide_au_all,
            name=f"tab_phase2_auroc_ALL_{score}",
            caption=f"Phase 2 AUROC (mean ± 95\\% CI half-width), combined tasks — {SCORER_PRETTY.get(score, score)}.",
            label=f"tab:phase2_auroc_all_{score}",
            out_dir=APPENDIX_DIR,
        )

        # Spearman combined
        wide_sp_all = build_all_task_table_per_scorer(
            df_sp,
            value_cols=("spearman_rho_boot_mean", "ci95_lo", "ci95_hi"),
            score=score,
            digits=DIGITS,
        )
        save_table_bundle(
            wide_sp_all,
            name=f"tab_phase2_spearman_ALL_{score}",
            caption=f"Phase 2 Spearman $\\rho$ (mean ± 95\\% CI half-width), combined tasks — {SCORER_PRETTY.get(score, score)}.",
            label=f"tab:phase2_spearman_all_{score}",
            out_dir=APPENDIX_DIR,
        )


if __name__ == "__main__":
    main()