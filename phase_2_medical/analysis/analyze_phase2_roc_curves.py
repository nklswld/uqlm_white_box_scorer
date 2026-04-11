"""Phase-2 scorer ROC analysis from archived per-example results.

This standalone script generates appendix-style scorer ROC overlays for the archived
Phase-2 baseline runs under ``phase_2_medical/outputs/final`` without rerunning any
models.

Appendix outputs:
- ``phase_2_medical/outputs/figures_tables/figures_general/fig_phase2_pubmedqa_roc_models.pdf``
- ``phase_2_medical/outputs/figures_tables/figures_general/fig_phase2_medqa_roc_models.pdf``

Design notes:
- score orientation follows the established Phase-2 convention: if a scorer yields
  AUROC < 0.5 on the relevant run, its sign is flipped so that higher values indicate
  higher error likelihood before ROC computation
- MTP is checked against LNTP on the archived PubMedQA runs; if it is numerically
  redundant, it is omitted from the main figure and this is documented in logging and
  PDF metadata
- the figure style mirrors the existing Phase-2 plotting defaults as closely as
  possible while remaining self-contained and reproducible
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_ROOT = REPO_ROOT / "phase_2_medical"
FINAL_DIR = PHASE2_ROOT / "outputs" / "final"
OUT_DIR = PHASE2_ROOT / "outputs" / "figures_tables" / "figures_general"

PUBMEDQA_FIG_PATH = OUT_DIR / "fig_phase2_pubmedqa_roc_models.pdf"
MEDQA_FIG_PATH = OUT_DIR / "fig_phase2_medqa_roc_models.pdf"

FONT_SCALE = 1.45
BASELINE_LINEWIDTH = 1.3
ROC_LINEWIDTH = 2.2
MTP_REDUNDANCY_TOL = 1e-12

MODEL_ORDER = ["mistral", "biomistral"]
MODEL_PRETTY = {"mistral": "Mistral", "biomistral": "BioMistral"}
TASK_PRETTY = {"pubmedqa": "PubMedQA", "medqa": "MedQA"}

SCORER_PRETTY = {
    "lntp": "LNTP",
    "mtp": "MTP",
    "egh_probe_oof": "EGH",
    "hidden_probe_oof": "Hidden",
}
SCORER_COLOR = {
    "lntp": "#4C72B0",
    "mtp": "#64B5CD",
    "egh_probe_oof": "#DD8452",
    "hidden_probe_oof": "#55A868",
}

PUBMEDQA_MAIN_SCORERS = ["lntp", "mtp", "egh_probe_oof", "hidden_probe_oof"]

PUBMEDQA_CAPTION_NOTE_WITH_MTP = (
    "Phase-2 scorer ROC curves for error discrimination on PubMedQA, shown separately "
    "for Mistral and BioMistral. Higher curves indicate better separation between "
    "incorrect and correct answers over threshold sweeps."
)
PUBMEDQA_CAPTION_NOTE_WITHOUT_MTP = (
    "Phase-2 scorer ROC curves for error discrimination on PubMedQA, shown separately "
    "for Mistral and BioMistral. Higher curves indicate better separation between "
    "incorrect and correct answers over threshold sweeps. MTP is omitted because it is "
    "numerically redundant with LNTP in the archived PubMedQA runs."
)
MEDQA_CAPTION_NOTE = (
    "Appendix-style Phase-2 scorer ROC curves for error discrimination on MedQA. The "
    "same scorer subset as the accompanying PubMedQA appendix figure is used for visual "
    "comparability."
)


@dataclass(frozen=True)
class RunSpec:
    """Canonical archived Phase-2 baseline run specification."""

    task: str
    model_family: str
    path: Path


@dataclass(frozen=True)
class RunData:
    """Loaded run artifact with labels and per-row scorer access."""

    task: str
    model_family: str
    path: Path
    rows: List[dict]
    y: np.ndarray
    label_key: str


@dataclass(frozen=True)
class RocSeries:
    """ROC-ready scorer data after orientation alignment."""

    scorer: str
    direction: float
    auroc: float
    n: int
    fpr: np.ndarray
    tpr: np.ndarray


RUN_SPECS: Sequence[RunSpec] = (
    RunSpec("pubmedqa", "mistral", FINAL_DIR / "pubmedqa_mistral.B5000.results.jsonl"),
    RunSpec("pubmedqa", "biomistral", FINAL_DIR / "pubmedqa_biomistral.B5000.results.jsonl"),
    RunSpec("medqa", "mistral", FINAL_DIR / "medqa_mistral.B5000.results.jsonl"),
    RunSpec("medqa", "biomistral", FINAL_DIR / "medqa_biomistral.B5000.results.jsonl"),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for optional appendix generation."""

    parser = argparse.ArgumentParser(
        description="Generate Phase-2 scorer ROC figures from archived result JSONL files."
    )
    parser.add_argument(
        "--skip_medqa_appendix",
        action="store_true",
        help="Skip the optional MedQA appendix-style ROC figure.",
    )
    return parser.parse_args()


def apply_phase2_plot_style(font_scale: float = FONT_SCALE) -> None:
    """Apply plotting defaults consistent with the existing Phase-2 scripts."""

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": int(12 * font_scale),
            "axes.titlesize": int(13.3 * font_scale),
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
    """Save a figure with the same locked-file fallback pattern used elsewhere."""

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(outpath, **kwargs)
        print(f"[OK] Saved: {outpath}")
        return outpath
    except PermissionError:
        for k in range(2, 50):
            alt = outpath.with_name(f"{outpath.stem}_v{k}{outpath.suffix}")
            try:
                fig.savefig(alt, **kwargs)
                print(f"[WARN] Locked -> wrote {alt}")
                return alt
            except PermissionError:
                continue
        raise


def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into a list of dictionaries."""

    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in results JSONL: {path}")
    return rows


def find_label_key(example: dict) -> str:
    """Infer the binary label field name from a single row."""

    for key in ["is_error", "label", "y", "target", "error"]:
        if key in example:
            return key
    raise KeyError("No label key found (expected one of is_error/label/y/target/error).")


def extract_scores(example: dict) -> Dict[str, float]:
    """Extract white-box scorer values from nested or flat result rows."""

    for key in ("scores", "wb_scores"):
        if key in example and isinstance(example[key], dict):
            return {
                str(score_name).lower(): float(score_value)
                for score_name, score_value in example[key].items()
                if isinstance(score_value, (int, float))
            }

    return {
        str(key).lower(): float(value)
        for key, value in example.items()
        if isinstance(value, (int, float))
        and any(token in str(key).lower() for token in ["lntp", "mtp", "egh", "hidden"])
    }


def load_run(spec: RunSpec) -> RunData:
    """Load and validate one archived baseline run."""

    rows = load_jsonl(spec.path)
    label_key = find_label_key(rows[0])
    y = np.asarray([int(row[label_key]) for row in rows], dtype=int)

    unique_y = set(np.unique(y).tolist())
    if not unique_y.issubset({0, 1}):
        raise ValueError(f"{spec.path.name} contains non-binary labels: {sorted(unique_y)}")
    if len(unique_y) < 2:
        raise ValueError(f"{spec.path.name} contains only one class; ROC is undefined.")

    return RunData(
        task=spec.task,
        model_family=spec.model_family,
        path=spec.path,
        rows=rows,
        y=y,
        label_key=label_key,
    )


def score_vector(rows: Sequence[dict], scorer: str) -> np.ndarray:
    """Return a scorer vector, filling missing row-level values with NaN."""

    values: List[float] = []
    for row in rows:
        scores = extract_scores(row)
        value = scores.get(scorer, np.nan)
        if value is None:
            value = np.nan
        values.append(float(value))
    return np.asarray(values, dtype=float)


def compute_roc_series(y: np.ndarray, raw_scores: np.ndarray, scorer: str) -> RocSeries:
    """Align score polarity to error likelihood and compute ROC arrays."""

    mask = np.isfinite(raw_scores)
    if int(mask.sum()) == 0:
        raise ValueError(f"{scorer}: no finite score values available.")

    y_use = y[mask]
    s_use = raw_scores[mask]
    if np.unique(y_use).size < 2:
        raise ValueError(f"{scorer}: only one class remains after finite-score filtering.")

    auroc = float(roc_auc_score(y_use, s_use))
    direction = +1.0
    if auroc < 0.5:
        s_use = -s_use
        auroc = float(roc_auc_score(y_use, s_use))
        direction = -1.0

    fpr, tpr, _ = roc_curve(y_use, s_use)
    return RocSeries(
        scorer=scorer,
        direction=direction,
        auroc=auroc,
        n=int(mask.sum()),
        fpr=fpr,
        tpr=tpr,
    )


def check_mtp_redundancy(pubmed_runs: Sequence[RunData]) -> dict:
    """Check whether MTP is numerically redundant with LNTP on PubMedQA."""

    details = []
    for run in pubmed_runs:
        lntp = score_vector(run.rows, "lntp")
        mtp = score_vector(run.rows, "mtp")
        mask = np.isfinite(lntp) & np.isfinite(mtp)
        if int(mask.sum()) == 0:
            details.append(
                {
                    "model": run.model_family,
                    "n_overlap": 0,
                    "max_abs_diff": np.nan,
                    "redundant": False,
                    "reason": "no finite overlap",
                }
            )
            continue

        diffs = np.abs(lntp[mask] - mtp[mask])
        max_abs_diff = float(np.max(diffs))
        redundant = bool(max_abs_diff <= MTP_REDUNDANCY_TOL)
        details.append(
            {
                "model": run.model_family,
                "n_overlap": int(mask.sum()),
                "max_abs_diff": max_abs_diff,
                "redundant": redundant,
                "reason": "exact overlap within tolerance" if redundant else "distinct values detected",
            }
        )

    omit_mtp = bool(details) and all(item["redundant"] for item in details)
    return {"omit_mtp": omit_mtp, "details": details}


def format_scorer_names(scorers: Sequence[str]) -> str:
    """Format scorer names for human-readable logging."""

    return ", ".join(SCORER_PRETTY.get(scorer, scorer) for scorer in scorers)


def plot_task_roc_figure(
    *,
    runs_by_model: Dict[str, RunData],
    task: str,
    scorers: Sequence[str],
    outpath: Path,
    title: str,
    caption_note: str,
) -> tuple[Path, Dict[str, List[str]]]:
    """Plot a two-panel ROC figure for the requested task."""

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), sharex=True, sharey=True)
    plotted_by_model: Dict[str, List[str]] = {}

    for ax, model_family in zip(axes, MODEL_ORDER):
        run = runs_by_model.get(model_family)
        if run is None:
            raise KeyError(f"Missing run for task={task}, model={model_family}")

        plotted_here: List[str] = []
        for scorer in scorers:
            raw_scores = score_vector(run.rows, scorer)
            try:
                roc_data = compute_roc_series(run.y, raw_scores, scorer=scorer)
            except ValueError as exc:
                print(
                    f"[WARN] Skipping scorer={scorer} for task={task}, "
                    f"model={model_family}: {exc}"
                )
                continue

            ax.plot(
                roc_data.fpr,
                roc_data.tpr,
                color=SCORER_COLOR.get(scorer, "tab:gray"),
                linewidth=ROC_LINEWIDTH,
                label=f"{SCORER_PRETTY.get(scorer, scorer)} (AUC={roc_data.auroc:.3f})",
            )
            plotted_here.append(scorer)
            print(
                f"[INFO] Plotted task={task}, model={model_family}, scorer={scorer}, "
                f"direction={roc_data.direction:+.0f}, AUROC={roc_data.auroc:.3f}, N={roc_data.n}"
            )

        ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=BASELINE_LINEWIDTH)
        ax.set_title(MODEL_PRETTY[model_family])
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="lower right", frameon=False, title="Scorer")
        plotted_by_model[model_family] = plotted_here

    axes[0].set_ylabel("True Positive Rate")
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    metadata = {"Title": title, "Subject": caption_note}
    saved_path = safe_savefig(fig, outpath, bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    return saved_path, plotted_by_model


def main() -> None:
    """Generate the requested Phase-2 ROC figures and print a concise summary."""

    args = parse_args()
    apply_phase2_plot_style()

    missing = [str(spec.path) for spec in RUN_SPECS if not spec.path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required archived result files: {missing}")

    loaded_runs = [load_run(spec) for spec in RUN_SPECS]
    pubmed_runs = [run for run in loaded_runs if run.task == "pubmedqa"]
    medqa_runs = [run for run in loaded_runs if run.task == "medqa"]

    pubmed_redundancy = check_mtp_redundancy(pubmed_runs)
    print("[INFO] PubMedQA LNTP vs MTP redundancy check:")
    for item in pubmed_redundancy["details"]:
        print(
            f"  - {MODEL_PRETTY[item['model']]}: n_overlap={item['n_overlap']}, "
            f"max_abs_diff={item['max_abs_diff']}, redundant={item['redundant']} "
            f"({item['reason']})"
        )

    pubmed_scorers = list(PUBMEDQA_MAIN_SCORERS)
    mtp_omitted = False
    if pubmed_redundancy["omit_mtp"]:
        pubmed_scorers = [scorer for scorer in pubmed_scorers if scorer != "mtp"]
        mtp_omitted = True
        print(
            "[INFO] MTP omitted from the main PubMedQA ROC figure because it is "
            "numerically redundant with LNTP in both archived PubMedQA runs."
        )
    else:
        print(
            "[INFO] MTP retained in the main PubMedQA ROC figure because it is not "
            "fully redundant with LNTP in the archived PubMedQA runs."
        )

    pubmed_runs_by_model = {run.model_family: run for run in pubmed_runs}
    pubmed_caption_note = (
        PUBMEDQA_CAPTION_NOTE_WITHOUT_MTP if mtp_omitted else PUBMEDQA_CAPTION_NOTE_WITH_MTP
    )
    saved_pubmed, pubmed_plotted = plot_task_roc_figure(
        runs_by_model=pubmed_runs_by_model,
        task="pubmedqa",
        scorers=pubmed_scorers,
        outpath=PUBMEDQA_FIG_PATH,
        title="Phase 2 ROC Curves on PubMedQA (Scorer-Based Error Discrimination)",
        caption_note=pubmed_caption_note,
    )

    saved_medqa = None
    medqa_plotted: Dict[str, List[str]] = {}
    if not args.skip_medqa_appendix:
        medqa_runs_by_model = {run.model_family: run for run in medqa_runs}
        saved_medqa, medqa_plotted = plot_task_roc_figure(
            runs_by_model=medqa_runs_by_model,
            task="medqa",
            scorers=pubmed_scorers,
            outpath=MEDQA_FIG_PATH,
            title="Phase 2 ROC Curves on MedQA (Appendix-Style Scorer Comparison)",
            caption_note=MEDQA_CAPTION_NOTE,
        )

    print("\n[SUMMARY] Phase-2 ROC analysis complete.")
    print(f"[SUMMARY] PubMedQA scorers plotted: {format_scorer_names(pubmed_scorers)}")
    print(f"[SUMMARY] MTP omitted as redundant: {'yes' if mtp_omitted else 'no'}")
    for model_family in MODEL_ORDER:
        plotted = format_scorer_names(pubmed_plotted.get(model_family, []))
        print(f"[SUMMARY] PubMedQA {MODEL_PRETTY[model_family]} panel: {plotted}")
    print(f"[SUMMARY] Saved figure: {saved_pubmed}")

    if saved_medqa is not None:
        print(f"[SUMMARY] MedQA scorers plotted: {format_scorer_names(pubmed_scorers)}")
        for model_family in MODEL_ORDER:
            plotted = format_scorer_names(medqa_plotted.get(model_family, []))
            print(f"[SUMMARY] MedQA {MODEL_PRETTY[model_family]} panel: {plotted}")
        print(f"[SUMMARY] Saved figure: {saved_medqa}")


if __name__ == "__main__":
    main()
