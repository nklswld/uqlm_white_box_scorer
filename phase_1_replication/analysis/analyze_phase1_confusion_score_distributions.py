"""Phase-1 post-hoc operating-point analysis for archived supervised OOF scorers.

This script complements the repository's AUROC-centric Phase-1 evaluation with a
descriptive, threshold-dependent analysis for the two strongest supervised scorers:

- ``scores.hidden_probe_oof``
- ``scores.egh_probe_oof``

Methodological note:
- the Youden-J threshold is computed on the same archived Phase-1 sample
- this is therefore a post-hoc operating-point analysis
- it complements AUROC and does not replace the main AUROC-based comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from phase1_figures import (
    apply_phase2_plot_style,
    ensure_dir,
    load_manifest,
    load_results_jsonl,
    safe_savefig,
)


LABEL_COL = "hallucinated"
GROUP_ORDER = ["TN", "FN", "FP", "TP"]
THRESHOLD_METHOD = "youden_j_same_sample_posthoc"

SCORERS: Dict[str, Dict[str, str]] = {
    "Hidden": {
        "results_col": "scores.hidden_probe_oof",
        "manifest_key": "hidden_probe_oof",
    },
    "EGH": {
        "results_col": "scores.egh_probe_oof",
        "manifest_key": "egh_probe_oof",
    },
}

GROUP_PALETTE = {
    "TN": "#4C72B0",
    "FN": "#E0A458",
    "FP": "#C44E52",
    "TP": "#55A868",
}

THRESHOLD_TEXT_FONTSIZE = 13
PANEL_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 13
SUPTITLE_FONTSIZE = 18


@dataclass(frozen=True)
class Args:
    """Typed CLI configuration for deterministic analysis exports."""

    results: Path
    manifest: Path
    outdir: Path
    font_scale: float


def parse_args() -> Args:
    """Parse CLI arguments and resolve canonical default paths."""
    script_dir = Path(__file__).resolve().parent
    phase1_root = script_dir.parent

    default_results = phase1_root / "outputs" / "phase1_truthfulqa_hallu_results_300.jsonl"
    default_manifest = phase1_root / "outputs" / "phase1_run_manifest.json"
    default_outdir = phase1_root / "outputs" / "figs"

    parser = argparse.ArgumentParser(
        description="Phase-1 post-hoc confusion-group score distribution analysis"
    )
    parser.add_argument("--results", type=str, default=str(default_results))
    parser.add_argument("--manifest", type=str, default=str(default_manifest))
    parser.add_argument("--outdir", type=str, default=str(default_outdir))
    parser.add_argument("--font_scale", type=float, default=1.25)

    parsed = parser.parse_args()
    return Args(
        results=Path(parsed.results).resolve(),
        manifest=Path(parsed.manifest).resolve(),
        outdir=Path(parsed.outdir).resolve(),
        font_scale=float(parsed.font_scale),
    )


def validate_inputs(results_path: Path, manifest_path: Path) -> None:
    """Fail fast when required input artifacts are missing."""
    if not results_path.exists():
        raise FileNotFoundError(f"Missing Phase-1 results JSONL: {results_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase-1 manifest JSON: {manifest_path}")


def load_phase1_artifacts(results_path: Path, manifest_path: Path) -> Tuple[pd.DataFrame, dict]:
    """Load the archived Phase-1 results and manifest artifacts."""
    validate_inputs(results_path, manifest_path)
    df = load_results_jsonl(results_path)
    manifest = load_manifest(manifest_path)
    return df, manifest


def prepare_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the required label/score columns into numeric form."""
    required_cols = [LABEL_COL] + [cfg["results_col"] for cfg in SCORERS.values()]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Results JSONL is missing required columns: {missing_cols}")

    out = df.copy()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    if out[LABEL_COL].isna().any():
        bad_n = int(out[LABEL_COL].isna().sum())
        raise ValueError(f"Label column '{LABEL_COL}' contains {bad_n} non-numeric values.")

    label_values = set(out[LABEL_COL].astype(int).unique().tolist())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Label column '{LABEL_COL}' must be binary 0/1, found: {sorted(label_values)}")

    out[LABEL_COL] = out[LABEL_COL].astype(int)

    for scorer_cfg in SCORERS.values():
        col = scorer_cfg["results_col"]
        out[col] = pd.to_numeric(out[col], errors="coerce")
        non_finite_mask = ~np.isfinite(out[col].to_numpy(dtype=float))
        if non_finite_mask.any():
            bad_n = int(non_finite_mask.sum())
            raise ValueError(
                f"Score column '{col}' contains {bad_n} non-finite values. "
                "This script expects archived finite OOF scores."
            )

    if out.empty:
        raise ValueError("Phase-1 results JSONL contains no rows.")

    return out


def validate_qid_column(df: pd.DataFrame) -> None:
    """Require a stable qid column for the long-format per-example audit export."""
    if "qid" not in df.columns:
        raise KeyError(
            "Results JSONL is missing required column 'qid'. "
            "The long-format export in this script is keyed by the archived example identifier."
        )


def get_manifest_auroc(manifest: dict, manifest_key: str) -> float:
    """Extract the authoritative scorer AUROC from the archived manifest."""
    try:
        value = float(manifest["scores"]["auroc"][manifest_key])
    except KeyError as exc:
        raise KeyError(f"Manifest is missing AUROC entry for '{manifest_key}'.") from exc

    if not np.isfinite(value):
        raise ValueError(f"Manifest AUROC for '{manifest_key}' is not finite: {value}")
    return value


def compute_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute the post-hoc Youden-J threshold on an archived score vector.

    Ties are broken deterministically by taking the first finite threshold among the
    maximal Youden-J operating points returned by ``roc_curve``.
    """
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)

    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and scores must have the same length.")
    if np.unique(y).size < 2:
        raise ValueError("Youden-J threshold is undefined when only one class is present.")
    if not np.all(np.isfinite(s)):
        raise ValueError("Youden-J threshold requires finite scores.")

    fpr, tpr, thresholds = roc_curve(y, s)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        raise ValueError("roc_curve produced no finite thresholds.")

    j_stat = tpr[finite_mask] - fpr[finite_mask]
    finite_thresholds = thresholds[finite_mask]
    best = np.flatnonzero(np.isclose(j_stat, j_stat.max()))
    if best.size == 0:
        raise ValueError("Failed to determine a Youden-J threshold.")

    return float(finite_thresholds[int(best[0])])


def assign_confusion_groups(
    base_df: pd.DataFrame,
    *,
    scorer_name: str,
    score_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Assign predictions and TP/FP/TN/FN groups for one scorer."""
    out = base_df.copy()
    y_true = out[LABEL_COL].to_numpy(dtype=int)
    score = out[score_col].to_numpy(dtype=float)
    y_pred = (score >= threshold).astype(int)

    conditions = [
        (y_true == 0) & (y_pred == 0),
        (y_true == 1) & (y_pred == 0),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 1),
    ]
    out["scorer"] = scorer_name
    out["score_column"] = score_col
    out["score"] = score
    out["threshold"] = float(threshold)
    out["predicted_label"] = y_pred
    out["group"] = np.select(conditions, GROUP_ORDER, default="UNASSIGNED")

    if (out["group"] == "UNASSIGNED").any():
        raise RuntimeError(f"Confusion-group assignment failed for scorer '{scorer_name}'.")

    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    return out


def ratio(num: int, den: int) -> float:
    """Safe ratio helper returning NaN for undefined denominators."""
    return float(num / den) if den > 0 else float("nan")


def build_operating_point_summary(
    assignments: pd.DataFrame,
    *,
    scorer_name: str,
    manifest_auroc: float,
) -> dict:
    """Compute scorer-level operating-point metrics from confusion assignments."""
    groups = assignments["group"].astype(str)
    tn = int((groups == "TN").sum())
    fn = int((groups == "FN").sum())
    fp = int((groups == "FP").sum())
    tp = int((groups == "TP").sum())

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    precision = ratio(tp, tp + fp)
    recall = sensitivity
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))

    balanced_accuracy = (
        float((sensitivity + specificity) / 2.0)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )

    labels = assignments[LABEL_COL].to_numpy(dtype=int)
    return {
        "scorer": scorer_name,
        "threshold_method": THRESHOLD_METHOD,
        "threshold": float(assignments["threshold"].iloc[0]),
        "manifest_auroc": float(manifest_auroc),
        "n_total": int(len(assignments)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def build_group_stats(assignments: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each scorer-by-group score distribution."""
    rows: List[dict] = []
    grouped = assignments.groupby(["scorer", "group"], observed=True)
    for (scorer_name, group), sub_df in grouped:
        values = sub_df["score"].to_numpy(dtype=float)
        rows.append(
            {
                "scorer": scorer_name,
                "group": str(group),
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No group-level statistics could be computed.")

    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    return out.sort_values(["scorer", "group"]).reset_index(drop=True)


def build_long_assignments_export(assignments_by_scorer: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Build the long-format per-example export used for thesis auditability."""
    frames = list(assignments_by_scorer)
    if not frames:
        raise ValueError("No scorer assignment frames were provided for long-format export.")

    validate_qid_column(frames[0])

    keep_cols = [
        "qid",
        LABEL_COL,
        "scorer",
        "score_column",
        "score",
        "threshold",
        "predicted_label",
        "group",
    ]
    frames = [df.loc[:, keep_cols].copy() for df in assignments_by_scorer]
    out = pd.concat(frames, axis=0, ignore_index=True)
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    return out.sort_values(["scorer", "group", "qid"]).reset_index(drop=True)


def format_group_labels(stats_df: pd.DataFrame, scorer_name: str) -> List[str]:
    """Create panel-specific x-axis labels with group counts."""
    sub = stats_df[stats_df["scorer"] == scorer_name].set_index("group")
    labels: List[str] = []
    for group in GROUP_ORDER:
        n = int(sub.loc[group, "n"]) if group in sub.index else 0
        labels.append(f"{group}\n(n={n})")
    return labels


def plot_confusion_group_distributions(
    assignments_long: pd.DataFrame,
    group_stats: pd.DataFrame,
    summary_df: pd.DataFrame,
    outpath: Path,
) -> None:
    """Render a publication-style combined 2-panel violin-plus-strip plot."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.8), sharey=True)

    for ax, scorer_name in zip(axes, SCORERS.keys()):
        sub = assignments_long[assignments_long["scorer"] == scorer_name].copy()
        sub["group"] = pd.Categorical(sub["group"], categories=GROUP_ORDER, ordered=True)

        sns.violinplot(
            data=sub,
            x="group",
            y="score",
            order=GROUP_ORDER,
            palette=GROUP_PALETTE,
            inner=None,
            cut=0,
            linewidth=1.0,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="group",
            y="score",
            order=GROUP_ORDER,
            color="black",
            alpha=0.42,
            jitter=0.18,
            size=3.0,
            ax=ax,
        )

        threshold = float(summary_df.loc[summary_df["scorer"] == scorer_name, "threshold"].iloc[0])
        ax.axhline(threshold, linestyle="--", linewidth=1.25, color="black", alpha=0.85)
        ax.text(
            -0.3,
            threshold + 0.04,
            f"Youden-J threshold\n= {threshold:.3f}",
            ha="left",
            va="bottom",
            fontsize=THRESHOLD_TEXT_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.25),
        )


        ax.set_title(scorer_name, fontsize=PANEL_TITLE_FONTSIZE)
        ax.set_xlabel("")
        ax.set_xticks(np.arange(len(GROUP_ORDER)))
        ax.set_xticklabels(format_group_labels(group_stats, scorer_name), fontsize=TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    axes[0].set_ylabel("OOF hallucination score", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel("")

    fig.suptitle(
        "Phase 1 confusion-group score distributions\n"
        "(post-hoc Youden-J operating points on archived OOF scores)",
        y=0.98,
        linespacing=1.45,
        fontsize=SUPTITLE_FONTSIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(wspace=0.24)
    safe_savefig(fig, outpath)
    plt.close(fig)


def main() -> None:
    """Run the full Phase-1 confusion-group operating-point analysis."""
    args = parse_args()

    df_raw, manifest = load_phase1_artifacts(args.results, args.manifest)
    df = prepare_results_frame(df_raw)

    sns.set_theme(style="white", rc={})
    apply_phase2_plot_style(font_scale=args.font_scale)
    ensure_dir(args.outdir)

    summary_rows: List[dict] = []
    assignments_frames: List[pd.DataFrame] = []

    for scorer_name, scorer_cfg in SCORERS.items():
        score_col = scorer_cfg["results_col"]
        manifest_auroc = get_manifest_auroc(manifest, scorer_cfg["manifest_key"])
        threshold = compute_youden_threshold(
            y_true=df[LABEL_COL].to_numpy(dtype=int),
            scores=df[score_col].to_numpy(dtype=float),
        )
        assignments = assign_confusion_groups(
            df,
            scorer_name=scorer_name,
            score_col=score_col,
            threshold=threshold,
        )
        assignments_frames.append(assignments)
        summary_rows.append(
            build_operating_point_summary(
                assignments,
                scorer_name=scorer_name,
                manifest_auroc=manifest_auroc,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["scorer"] = pd.Categorical(summary_df["scorer"], categories=list(SCORERS.keys()), ordered=True)
    summary_df = summary_df.sort_values("scorer").reset_index(drop=True)

    assignments_long = pd.concat(assignments_frames, axis=0, ignore_index=True)
    assignments_long["group"] = pd.Categorical(assignments_long["group"], categories=GROUP_ORDER, ordered=True)
    assignments_long = assignments_long.sort_values(["scorer", "group"]).reset_index(drop=True)
    group_stats_df = build_group_stats(assignments_long)

    figure_path = args.outdir / "fig_phase1_confusion_groups_hidden_egh.pdf"

    plot_confusion_group_distributions(
        assignments_long=assignments_long,
        group_stats=group_stats_df,
        summary_df=summary_df,
        outpath=figure_path,
    )

    print(f"[OK] Wrote: {figure_path}")

    for row in summary_df.to_dict(orient="records"):
        print(
            "[SUMMARY] "
            f"{row['scorer']}: "
            f"threshold={row['threshold']:.3f}, "
            f"TN/FP/FN/TP={int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tp'])}, "
            f"precision={row['precision']:.3f}, "
            f"recall={row['recall']:.3f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
