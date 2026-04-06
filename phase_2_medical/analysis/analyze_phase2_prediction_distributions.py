"""Descriptive Phase-2 diagnostic analysis of model prediction distributions.

This script uses the archived canonical Phase-2 baseline result artifacts to inspect
whether Mistral and BioMistral exhibit different predicted-class distributions and
confusion structures, with primary emphasis on PubMedQA and the `yes`/`no`/`maybe`
label space.

Interpretation note:
- this analysis is descriptive only
- it can help assess whether prediction-distribution shifts plausibly contribute to
  model-dependent scorer differences
- it does not establish a causal explanation for AUROC or Spearman differences
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from phase2_tables import load_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = REPO_ROOT / "phase_2_medical"
FINAL_DIR = PHASE2_ROOT / "outputs" / "final"
TABLES_DIR = PHASE2_ROOT / "outputs" / "figures_tables" / "tables_general"
FIGS_DIR = PHASE2_ROOT / "outputs" / "figures_tables" / "figures_general"

DIST_CSV_NAME = "analysis_phase2_prediction_class_distributions.csv"
METRICS_CSV_NAME = "analysis_phase2_prediction_per_class_metrics.csv"
PUBMED_DIST_FIG_NAME = "fig_phase2_pubmedqa_predicted_class_distribution_models.pdf"
PUBMED_CM_FIG_NAME = "fig_phase2_pubmedqa_confusion_matrices_models.pdf"

PUBMEDQA_ORDER = ["yes", "no", "maybe"]
MEDQA_ORDER = ["A", "B", "C", "D"]
MISSING_PRED_LABEL = "__MISSING__"

MODEL_COLOR = {"mistral": "tab:blue", "biomistral": "tab:orange"}
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}
TASK_PRETTY = {"pubmedqa": "PubMedQA", "medqa": "MedQA"}


@dataclass(frozen=True)
class RunSpec:
    """Canonical archived Phase-2 baseline run specification."""

    task: str
    model_family: str
    path: Path


RUN_SPECS: Sequence[RunSpec] = (
    RunSpec("pubmedqa", "mistral", FINAL_DIR / "pubmedqa_mistral.B5000.results.jsonl"),
    RunSpec("pubmedqa", "biomistral", FINAL_DIR / "pubmedqa_biomistral.B5000.results.jsonl"),
    RunSpec("medqa", "mistral", FINAL_DIR / "medqa_mistral.B5000.results.jsonl"),
    RunSpec("medqa", "biomistral", FINAL_DIR / "medqa_biomistral.B5000.results.jsonl"),
)


@dataclass(frozen=True)
class Args:
    """Typed CLI configuration for deterministic Phase-2 diagnostics."""

    out_tables_dir: Path
    out_figs_dir: Path


def parse_args() -> Args:
    """Parse CLI arguments for output locations."""
    parser = argparse.ArgumentParser(
        description="Phase-2 diagnostic analysis of model prediction distributions"
    )
    parser.add_argument("--out_tables_dir", type=str, default=str(TABLES_DIR))
    parser.add_argument("--out_figs_dir", type=str, default=str(FIGS_DIR))
    parsed = parser.parse_args()

    return Args(
        out_tables_dir=Path(parsed.out_tables_dir).resolve(),
        out_figs_dir=Path(parsed.out_figs_dir).resolve(),
    )


def apply_phase2_plot_style(font_scale: float = 1.25) -> None:
    """Apply publication-style plotting defaults consistent with existing Phase-2 figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": int(12 * font_scale),
            "axes.titlesize": int(13 * font_scale),
            "axes.labelsize": int(12 * font_scale),
            "xtick.labelsize": int(11 * font_scale),
            "ytick.labelsize": int(11 * font_scale),
            "legend.fontsize": int(11 * font_scale),
            "legend.title_fontsize": int(11 * font_scale),
            "figure.titlesize": int(14.8 * font_scale),
            "axes.titlepad": 12,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.minor.size": 3.0,
            "ytick.minor.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def safe_savefig(fig: plt.Figure, outpath: Path, **kwargs) -> Path:
    """Save a figure with the same locked-file fallback pattern used in Phase-2 figures."""
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        return outpath
    except PermissionError:
        for k in range(2, 50):
            alt = outpath.with_name(f"{outpath.stem}_v{k}{outpath.suffix}")
            try:
                fig.savefig(alt, **kwargs)
                print(f"[WARN] Locked -> wrote {alt.name}")
                return alt
            except PermissionError:
                continue
        raise


def ensure_exists(paths: Iterable[Path]) -> None:
    """Fail fast if any canonical archived input artifact is missing."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Phase-2 input files: {missing}")


def label_order_for_task(task: str) -> List[str]:
    """Return the canonical class order for a task."""
    task_l = str(task).lower()
    if task_l == "pubmedqa":
        return PUBMEDQA_ORDER.copy()
    if task_l == "medqa":
        return MEDQA_ORDER.copy()
    raise ValueError(f"Unsupported task: {task}")


def canonicalize_label(value: object, task: str) -> str | None:
    """Normalize gold/pred labels into the task-specific canonical label space."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if task == "pubmedqa":
        return text.lower()
    if task == "medqa":
        return text.upper()
    raise ValueError(f"Unsupported task: {task}")


def load_run(spec: RunSpec) -> pd.DataFrame:
    """Load and validate a single canonical Phase-2 baseline run."""
    rows = load_jsonl(spec.path)
    if not rows:
        raise ValueError(f"No rows found in results JSONL: {spec.path}")

    df = pd.json_normalize(rows)
    required_cols = ["qid", "task", "gold", "pred", "label", "meta.model"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"{spec.path.name} is missing required columns: {missing}")

    df = df.copy()
    df["task"] = df["task"].astype(str).str.lower()
    if set(df["task"].unique()) != {spec.task}:
        raise ValueError(
            f"{spec.path.name} expected task '{spec.task}', found tasks {sorted(df['task'].unique().tolist())}"
        )

    df["gold"] = df["gold"].map(lambda x: canonicalize_label(x, spec.task))
    df["pred"] = df["pred"].map(lambda x: canonicalize_label(x, spec.task))
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    if df["label"].isna().any():
        raise ValueError(f"{spec.path.name} contains non-numeric `label` values.")
    df["label"] = df["label"].astype(int)

    valid_labels = set(label_order_for_task(spec.task))
    gold_bad = sorted({x for x in df["gold"].dropna().unique().tolist() if x not in valid_labels})
    pred_bad = sorted({x for x in df["pred"].dropna().unique().tolist() if x not in valid_labels})
    if gold_bad:
        raise ValueError(f"{spec.path.name} contains unexpected gold labels: {gold_bad}")
    if pred_bad:
        raise ValueError(f"{spec.path.name} contains unexpected pred labels: {pred_bad}")
    if not set(df["label"].unique().tolist()).issubset({0, 1}):
        raise ValueError(f"{spec.path.name} contains non-binary `label` values.")

    model_names = sorted(df["meta.model"].dropna().astype(str).unique().tolist())
    model_name = model_names[0] if model_names else spec.model_family
    if len(model_names) > 1:
        raise ValueError(f"{spec.path.name} contains multiple model names: {model_names}")

    df["model_family"] = spec.model_family
    df["model_name"] = model_name
    df["source_file"] = spec.path.name
    return df


def build_distribution_rows(df: pd.DataFrame) -> List[dict]:
    """Compute gold/pred class counts and proportions for a single run."""
    task = str(df["task"].iloc[0]).lower()
    model_family = str(df["model_family"].iloc[0]).lower()
    model_name = str(df["model_name"].iloc[0])
    order = label_order_for_task(task)
    n_total = int(len(df))

    rows: List[dict] = []
    for distribution_type, series in (("gold", df["gold"]), ("pred", df["pred"])):
        counts = series.value_counts(dropna=False)
        labels = order.copy()
        missing_pred_n = int(series.isna().sum())
        if distribution_type == "pred" and missing_pred_n > 0:
            labels.append(MISSING_PRED_LABEL)

        for class_label in labels:
            if class_label == MISSING_PRED_LABEL:
                count = missing_pred_n
            else:
                count = int(counts.get(class_label, 0))
            rows.append(
                {
                    "task": task,
                    "model_family": model_family,
                    "model_name": model_name,
                    "class_label": class_label,
                    "distribution_type": distribution_type,
                    "count": count,
                    "proportion": float(count / n_total),
                    "n_total": n_total,
                }
            )

    return rows


def safe_div(num: int, den: int) -> float:
    """Return a safe floating-point division, or NaN when undefined."""
    return float(num / den) if den > 0 else float("nan")


def build_per_class_metric_rows(df: pd.DataFrame) -> List[dict]:
    """Compute per-class precision/recall/F1 and overall accuracy for a run."""
    task = str(df["task"].iloc[0]).lower()
    model_family = str(df["model_family"].iloc[0]).lower()
    model_name = str(df["model_name"].iloc[0])
    order = label_order_for_task(task)

    gold = df["gold"]
    pred = df["pred"]
    overall_accuracy = float((gold == pred).fillna(False).mean())
    missing_pred_n = int(pred.isna().sum())

    rows: List[dict] = []
    for class_label in order:
        tp = int(((gold == class_label) & (pred == class_label)).sum())
        fp = int(((gold != class_label) & (pred == class_label)).sum())
        fn = int(((gold == class_label) & (pred != class_label)).sum())
        support = int((gold == class_label).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, support)
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan")
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))

        rows.append(
            {
                "task": task,
                "model_family": model_family,
                "model_name": model_name,
                "class_label": class_label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy_overall": overall_accuracy,
                "missing_pred_n": missing_pred_n,
                "n_total": int(len(df)),
            }
        )

    return rows


def confusion_counts(df: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Build a task-specific confusion count matrix with fixed label order."""
    gold = pd.Categorical(df["gold"], categories=list(order), ordered=True)
    pred = pd.Categorical(df["pred"], categories=list(order), ordered=True)
    cm = pd.crosstab(gold, pred, dropna=False)
    return cm.reindex(index=order, columns=order, fill_value=0)


def plot_pubmedqa_distribution(dist_df: pd.DataFrame, outpath: Path) -> None:
    """Plot PubMedQA predicted-class proportions for Mistral vs BioMistral."""
    plot_df = dist_df[
        (dist_df["task"] == "pubmedqa") &
        (dist_df["distribution_type"] == "pred") &
        (dist_df["class_label"].isin(PUBMEDQA_ORDER))
    ].copy()
    plot_df["class_label"] = pd.Categorical(plot_df["class_label"], categories=PUBMEDQA_ORDER, ordered=True)
    plot_df["model_family"] = pd.Categorical(plot_df["model_family"], categories=["mistral", "biomistral"], ordered=True)
    plot_df = plot_df.sort_values(["class_label", "model_family"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.barplot(
        data=plot_df,
        x="class_label",
        y="proportion",
        hue="model_family",
        palette=MODEL_COLOR,
        ax=ax,
    )

    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        x = patch.get_x() + patch.get_width() / 2.0
        y = patch.get_height()
        ax.text(
            x,
            y + 0.012,
            f"{row['proportion']:.2f}\n(n={int(row['count'])})",
            ha="center",
            va="bottom",
            fontsize=int(mpl.rcParams["xtick.labelsize"] * 0.9),
        )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0.0, max(0.85, float(plot_df["proportion"].max()) + 0.12))
    ax.set_xticklabels(["yes", "no", "maybe"])
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [MODEL_PRETTY["mistral"], MODEL_PRETTY["biomistral"]], title="Model", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    fig.suptitle("PubMedQA predicted-class distribution by model", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)


def plot_pubmedqa_confusion_matrices(run_dfs: Dict[str, pd.DataFrame], outpath: Path) -> None:
    """Plot side-by-side PubMedQA confusion matrices for Mistral and BioMistral."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=True, sharey=True)
    vmax = 0
    cms: Dict[str, pd.DataFrame] = {}
    for model_family in ("mistral", "biomistral"):
        cm = confusion_counts(run_dfs[model_family], PUBMEDQA_ORDER)
        cms[model_family] = cm
        vmax = max(vmax, int(cm.to_numpy().max()))

    for ax, model_family in zip(axes, ("mistral", "biomistral")):
        sns.heatmap(
            cms[model_family],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            vmin=0,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(MODEL_PRETTY[model_family])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Gold label")

    fig.suptitle("PubMedQA confusion matrices by model", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(wspace=-0.25)
    safe_savefig(fig, outpath, bbox_inches="tight")
    plt.close(fig)


def summarize_findings(dist_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Print concise console summaries of the most relevant descriptive findings."""
    pub_pred = dist_df[
        (dist_df["task"] == "pubmedqa") &
        (dist_df["distribution_type"] == "pred") &
        (dist_df["class_label"].isin(PUBMEDQA_ORDER))
    ].copy()
    pub_pred = pub_pred.set_index(["model_family", "class_label"])

    mistral_maybe = float(pub_pred.loc[("mistral", "maybe"), "proportion"])
    biomistral_maybe = float(pub_pred.loc[("biomistral", "maybe"), "proportion"])
    mistral_yes = float(pub_pred.loc[("mistral", "yes"), "proportion"])
    biomistral_yes = float(pub_pred.loc[("biomistral", "yes"), "proportion"])

    pub_acc = metrics_df[metrics_df["task"] == "pubmedqa"][["model_family", "accuracy_overall"]].drop_duplicates()
    med_acc = metrics_df[metrics_df["task"] == "medqa"][["model_family", "accuracy_overall"]].drop_duplicates()

    print("[INFO] PubMedQA predicted-class proportions:")
    for model_family in ("mistral", "biomistral"):
        parts = []
        for cls in PUBMEDQA_ORDER:
            row = pub_pred.loc[(model_family, cls)]
            parts.append(f"{cls}={row['proportion']:.3f} (n={int(row['count'])})")
        print(f"  {MODEL_PRETTY[model_family]}: " + ", ".join(parts))

    print("[INFO] Overall accuracy by task/model:")
    for _, row in pd.concat([pub_acc.assign(task="pubmedqa"), med_acc.assign(task="medqa")]).iterrows():
        print(f"  {TASK_PRETTY[row['task']]} × {MODEL_PRETTY[row['model_family']]}: {float(row['accuracy_overall']):.3f}")

    print("[INFO] PubMedQA descriptive takeaway:")
    print(
        "  "
        f"BioMistral predicts `maybe` less often than Mistral "
        f"({biomistral_maybe:.3f} vs {mistral_maybe:.3f}) and predicts `yes` more often "
        f"({biomistral_yes:.3f} vs {mistral_yes:.3f})."
    )


def main() -> None:
    """Run the archived Phase-2 prediction-distribution diagnostic analysis."""
    args = parse_args()
    ensure_exists([spec.path for spec in RUN_SPECS])

    sns.set_theme(style="white", rc={})
    apply_phase2_plot_style(font_scale=1.2)

    args.out_tables_dir.mkdir(parents=True, exist_ok=True)
    args.out_figs_dir.mkdir(parents=True, exist_ok=True)

    run_frames = [load_run(spec) for spec in RUN_SPECS]
    all_df = pd.concat(run_frames, axis=0, ignore_index=True)

    distribution_rows: List[dict] = []
    metric_rows: List[dict] = []
    for _, run_df in all_df.groupby(["task", "model_family"], sort=False):
        distribution_rows.extend(build_distribution_rows(run_df))
        metric_rows.extend(build_per_class_metric_rows(run_df))

    dist_df = pd.DataFrame(distribution_rows)
    metrics_df = pd.DataFrame(metric_rows)

    dist_df.to_csv(args.out_tables_dir / DIST_CSV_NAME, index=False)
    metrics_df.to_csv(args.out_tables_dir / METRICS_CSV_NAME, index=False)

    pubmedqa_runs = {
        model_family: all_df[(all_df["task"] == "pubmedqa") & (all_df["model_family"] == model_family)].copy()
        for model_family in ("mistral", "biomistral")
    }
    plot_pubmedqa_distribution(dist_df, args.out_figs_dir / PUBMED_DIST_FIG_NAME)
    plot_pubmedqa_confusion_matrices(pubmedqa_runs, args.out_figs_dir / PUBMED_CM_FIG_NAME)

    print(f"[OK] Wrote: {args.out_tables_dir / DIST_CSV_NAME}")
    print(f"[OK] Wrote: {args.out_tables_dir / METRICS_CSV_NAME}")
    print(f"[OK] Wrote: {args.out_figs_dir / PUBMED_DIST_FIG_NAME}")
    print(f"[OK] Wrote: {args.out_figs_dir / PUBMED_CM_FIG_NAME}")

    for _, run_df in all_df.groupby(["task", "model_family"], sort=False):
        task = str(run_df["task"].iloc[0]).lower()
        model_family = str(run_df["model_family"].iloc[0]).lower()
        model_name = str(run_df["model_name"].iloc[0])
        missing_pred_n = int(run_df["pred"].isna().sum())
        print(
            "[RUN] "
            f"{TASK_PRETTY[task]} × {MODEL_PRETTY[model_family]} | "
            f"model_name={model_name} | n={len(run_df)} | missing_pred={missing_pred_n}"
        )

    summarize_findings(dist_df, metrics_df)


if __name__ == "__main__":
    main()
