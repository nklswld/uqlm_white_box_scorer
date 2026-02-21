"""Phase 2 table generation for medical error-detection metrics.

Reads aggregated Phase 2 metrics (bootstrap mean and 95% CI bounds) from CSV files
and emits publication-ready wide tables for per-task comparisons and appendix-style
combined-task views (per scorer).
Inputs: metrics CSVs under outputs/final (AUROC and Spearman rho summaries).
Outputs: CSV tables under outputs/tables and outputs/tables/appendix.
Determinism: deterministic given fixed CSV contents; no randomness or sampling here.
"""

# phase_2_medical/analysis/phase2_tables.py 
import pandas as pd
from pathlib import Path
import numpy as np


# ----------------------------
# Paths
# ----------------------------
# Base directory for upstream Phase 2 metric summaries (produced elsewhere).
BASE = (Path(__file__).resolve().parents[1] / "outputs" / "final")
# Output directory for generated table CSVs.
OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs" / "tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Appendix outputs: "combined" tables spanning tasks for a fixed scorer.
APPENDIX_DIR = OUT_DIR / "appendix"
APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

# Metric summary inputs (must contain task/model/score keys plus metric + CI columns).
AUROC_CSV = BASE / "phase2_metrics_auroc_ci_filtered.csv"
SPEAR_CSV = BASE / "phase2_metrics_spearman_rho_filtered.csv"


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
    """Return a compact 'mean ± halfwidth' CI string; missing values render as em dash."""
    if any(pd.isna(x) for x in [mean, lo, hi]):
        return "—"
    # CI is provided as [lo, hi]; we report half-width to match manuscript convention.
    pm = (hi - lo) / 2.0
    return f"{mean:.{digits}f} ± {pm:.{digits}f}"

def _get_row(
    df: pd.DataFrame,
    task: str,
    model: str,
    score: str,
    prefer_col: str | None = None,
) -> pd.Series | None:
    r = df[(df["task"] == task) & (df["model"] == model) & (df["score"] == score)]
    if len(r) == 0:
        return None

    if len(r) > 1 and prefer_col is not None:
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
    """
    Build a per-task wide table with scorers as rows and models as columns.
    Cells are formatted as 'mean ± CI half-width' (or '—' if missing).
    """
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for score in SCORER_ORDER:
        row = {"Scorer": SCORER_PRETTY.get(score, score)}
        for model in MODEL_ORDER:
            # Alignment invariant: one cell corresponds to a unique (task, model, score) triple.
            r = _get_row(df, task, model, score, prefer_col=mean_col)
            if r is None:
                # Missing combinations are rendered explicitly to avoid silent table shape changes.
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
    """Write a single CSV table to disk (LaTeX assets handled elsewhere)."""
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
    """
    Build a per-scorer wide table with tasks as rows and models as columns.
    Intended for appendix: highlights cross-task differences for a fixed scoring method.
    """
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for task in TASK_ORDER:
        row = {"Task": TASK_PRETTY.get(task, task)}
        for model in MODEL_ORDER:
            # Alignment invariant: one cell corresponds to a unique (task, model, score) triple.
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
# Main
# ----------------------------
def main():
    """Generate per-task and appendix combined-task tables for AUROC and Spearman rho."""
    # --- AUROC ---
    if not AUROC_CSV.exists():
        raise FileNotFoundError(f"Missing AUROC CSV: {AUROC_CSV}")
    df_au = pd.read_csv(AUROC_CSV)
    
    for k in ["task", "model", "score"]:
        df_au[k] = df_au[k].astype(str).str.lower()
    
    # --- column aliasing (phase2_figures.py writes 'auroc_boot_mean', tables expect 'boot_mean')
    if "boot_mean" not in df_au.columns:
        if "auroc_boot_mean" in df_au.columns:
            df_au = df_au.rename(columns={"auroc_boot_mean": "boot_mean"})
        elif "auroc" in df_au.columns:
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