# phase_2_medical/analysis/phase2_tables.py
import pandas as pd
from pathlib import Path
import numpy as np


# ----------------------------
# Paths
# ----------------------------
BASE = (Path(__file__).resolve().parents[1] / "outputs" / "final")
OUT_DIR = (Path(__file__).resolve().parents[1] / "outputs" / "tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NEW (besprochen): Combined-Tabellen in Appendix-Unterordner
APPENDIX_DIR = OUT_DIR / "appendix"
APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

AUROC_CSV = BASE / "phase2_metrics_auroc_ci_filtered.csv"
SPEAR_CSV = BASE / "phase2_metrics_spearman_rho_filtered.csv"


# ----------------------------
# Config
# ----------------------------
TASK_ORDER = ["medqa", "pubmedqa"]
MODEL_ORDER = ["mistral", "biomistral"]
SCORER_ORDER = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]

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

# NEW (besprochen): Task-Labels für Combined-Tabellen
TASK_PRETTY = {
    "medqa": "MedQA",
    "pubmedqa": "PubMedQA",
}

DIGITS = 2


# ----------------------------
# Helpers
# ----------------------------
def mean_ci_str(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    """Format as mean ± halfwidth of CI."""
    if any(pd.isna(x) for x in [mean, lo, hi]):
        return "—"
    pm = (hi - lo) / 2.0
    return f"{mean:.{digits}f} ± {pm:.{digits}f}"


def _get_row(df: pd.DataFrame, task: str, model: str, score: str) -> pd.Series | None:
    r = df[(df["task"] == task) & (df["model"] == model) & (df["score"] == score)]
    if len(r) == 0:
        return None
    # If duplicates exist, keep first but warn via comment / consistent behavior
    return r.iloc[0]


def build_wide_table(
    df: pd.DataFrame,
    value_cols: tuple[str, str, str],  # (mean_col, lo_col, hi_col)
    task: str,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Returns a wide table:
        Scorer | Mistral | BioMistral
    Each cell formatted as "mean ± pm".
    """
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for score in SCORER_ORDER:
        row = {"Scorer": SCORER_PRETTY.get(score, score)}
        for model in MODEL_ORDER:
            r = _get_row(df, task, model, score)
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



# Optional out_dir, damit Combined in Appendix landet
def save_table_bundle(wide: pd.DataFrame, name: str, caption: str, label: str, out_dir: Path = OUT_DIR):
    """
    Save:
      - CSV only
    """
    csv_path = out_dir / f"{name}.csv"
    wide.to_csv(csv_path, index=False)

    print(f"[OK] Wrote: {csv_path}")


# Combined ALL-Task Table pro Scorer (für Anhang)
def build_all_task_table_per_scorer(
    df: pd.DataFrame,
    value_cols: tuple[str, str, str],  # (mean_col, lo_col, hi_col)
    score: str,
    digits: int = 3,
) -> pd.DataFrame:
    """
    Combined table for one scorer (for appendix):
        Task | Mistral | BioMistral
    """
    mean_col, lo_col, hi_col = value_cols

    rows = []
    for task in TASK_ORDER:
        row = {"Task": TASK_PRETTY.get(task, task)}
        for model in MODEL_ORDER:
            r = _get_row(df, task, model, score)
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
    # --- AUROC ---
    if not AUROC_CSV.exists():
        raise FileNotFoundError(f"Missing AUROC CSV: {AUROC_CSV}")
    df_au = pd.read_csv(AUROC_CSV)

    # expected columns: boot_mean, ci95_lo, ci95_hi
    for c in ["task", "model", "score", "boot_mean", "ci95_lo", "ci95_hi"]:
        if c not in df_au.columns:
            raise KeyError(f"AUROC CSV missing column '{c}'. Found: {list(df_au.columns)}")

    # --- Spearman ---
    if not SPEAR_CSV.exists():
        raise FileNotFoundError(f"Missing Spearman CSV: {SPEAR_CSV}")
    df_sp = pd.read_csv(SPEAR_CSV)

    # expected columns: spearman_rho_boot_mean, ci95_lo, ci95_hi
    for c in ["task", "model", "score", "spearman_rho_boot_mean", "ci95_lo", "ci95_hi"]:
        if c not in df_sp.columns:
            raise KeyError(f"Spearman CSV missing column '{c}'. Found: {list(df_sp.columns)}")

    # build per-task tables
    tasks = [t for t in TASK_ORDER if t in set(df_au["task"].unique()) or t in set(df_sp["task"].unique())]
    if not tasks:
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

    # Combined Tabellen -> Appendix
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