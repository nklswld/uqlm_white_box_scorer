# src/run_phase1_truthfulqa.py
from __future__ import annotations  # NOTE: Enables forward refs + cleaner type hints.

import argparse  # NOTE: CLI interface for reproducible experiment runs.
import json  # NOTE: Reading/writing JSONL + manifest.
import logging  # NOTE: Structured logs for auditability.
import os  # NOTE: Environment configuration (Torch runtime flags, seeds).
import platform  # NOTE: System info for manifest.
import sys  # NOTE: Python version + runtime metadata for manifest.
import time  # NOTE: Timestamping output artifacts.
from dataclasses import dataclass, asdict  # NOTE: Typed config/results serialization.
from pathlib import Path  # NOTE: Stable file path handling across OS.
from typing import Any, Dict, List, Tuple  # NOTE: Type hints for clarity.

import numpy as np  # NOTE: Core numerical operations + arrays.


def configure_torch_runtime() -> None:
    # NOTE: Runtime flags that can affect speed/memory determinism.
    os.environ.setdefault("TORCH_SDPA_ENABLE", "1")
    os.environ.setdefault("TORCH_SDPA_DISABLE", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    try:
        import torch

        # NOTE: Enable fast SDPA backends when CUDA is available (best-effort).
        if torch.cuda.is_available():
            cuda_backends = getattr(torch.backends, "cuda", None)
            if cuda_backends is not None:
                for fn_name, arg in (
                    ("enable_flash_sdp", True),
                    ("enable_mem_efficient_sdp", True),
                    ("enable_math_sdp", False),
                ):
                    fn = getattr(cuda_backends, fn_name, None)
                    if callable(fn):
                        fn(arg)
    except Exception:
        # NOTE: Defensive: do not crash if runtime flags differ by torch version.
        pass


configure_torch_runtime()

import torch  # noqa: E402  # NOTE: Imported after environment runtime configuration.
import transformers  # noqa: E402  (Should-Fix A)  # NOTE: Version captured in manifest.
import sklearn  # noqa: E402        (Should-Fix A)  # NOTE: Version captured in manifest.
from sklearn.metrics import roc_auc_score  # noqa: E402  # NOTE: AUROC metric.

from modeling_llm import LLMWrapper  # noqa: E402  # NOTE: Unified model/tokenizer wrapper.
from scorers_logit import compute_lntp_mtp_for_qa_batch  # noqa: E402  # NOTE: LNTP/MTP scorer.
from scorers_gradient import compute_egh_primitives_for_qa  # noqa: E402  # NOTE: EGH primitives.
from scorers_hidden import build_hidden_feature_matrix, HiddenFeatureConfig, oof_logreg_scores  # noqa: E402
# NOTE: Hidden-state feature extraction + OOF probe training.
from bootstrap import BootstrapConfig, bootstrap_auc, bootstrap_auc_diff_with_indices  # noqa: E402
# NOTE: Bootstrap CI + delta-vs-random with stored indices.


def setup_logging(verbosity: int) -> None:
    # NOTE: Keep logging predictable across runs and verbosity levels.
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


logger = logging.getLogger(__name__)  # NOTE: Module-level logger.


def set_global_seeds(seed: int) -> None:
    # NOTE: Seed all relevant RNGs for reproducibility.
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: Determinism flags (can impact speed).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    # NOTE: Minimal safe AUROC wrapper with sanity checks.
    y = np.asarray(y).reshape(-1).astype(int)
    s = np.asarray(s).reshape(-1).astype(float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("AUROC: y and scores must have same length.")
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: only one class present.")
    return float(roc_auc_score(y, s))


def bootstrap_result_to_dict(res: Any) -> Dict[str, Any]:
    # NOTE: Normalize bootstrap result objects into JSON-serializable dicts.
    try:
        return asdict(res)
    except Exception:
        pass

    if hasattr(res, "__dict__"):
        return dict(res.__dict__)

    out: Dict[str, Any] = {}
    for k in ("auc", "ci_low", "ci_high", "B", "ci", "seed", "stratified"):
        if hasattr(res, k):
            out[k] = getattr(res, k)
    return out


def _jsonify(obj: Any) -> Any:
    # NOTE: Convert NumPy/scalars/containers into plain JSON-friendly types.
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def ensure_parent_dir(path: Path) -> None:
    # NOTE: Ensure output directory exists.
    path.parent.mkdir(parents=True, exist_ok=True)


def die(msg: str, code: int = 2) -> None:
    # NOTE: Centralized fatal exit with logging.
    logger.error(msg)
    raise SystemExit(code)


@dataclass(frozen=True)
class TruthfulQAExample:
    # NOTE: Minimal record for frozen-answer evaluation.
    qid: str
    question: str
    model_answer: str
    label: int  # 1 = hallucinated, 0 = non-hallucinated


def load_truthfulqa_examples(path: Path) -> List[TruthfulQAExample]:
    # NOTE: Load JSONL of frozen outputs + labels.
    examples: List[TruthfulQAExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            examples.append(
                TruthfulQAExample(
                    qid=str(obj["qid"]),
                    question=str(obj["question"]),
                    model_answer=str(obj["model_answer"]),
                    label=int(obj["hallucinated"]),
                )
            )
    return examples


@dataclass(frozen=True)
class Phase1Config:
    # NOTE: All experiment knobs are explicitly represented here for reproducibility.
    input_jsonl: Path
    output_jsonl: Path
    output_manifest: Path

    model_name: str
    device: str
    torch_dtype: str

    seed: int
    B: int
    ci: float
    n_splits: int

    hidden_layers: Tuple[int, ...]
    hidden_pooling: str
    hidden_normalize: bool

    batch_size: int


def parse_args() -> Phase1Config:
    # NOTE: Resolve paths relative to repository /phase_1_replication root.
    base_dir = Path(__file__).resolve().parents[1]  # .../phase_1_replication

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (base_dir / pp)

    # NOTE: CLI args define the full experimental protocol (inputs, outputs, model, seeds).
    p = argparse.ArgumentParser(description="Phase 1 – TruthfulQA Hallucination Detection (Frozen Outputs)")
    p.add_argument("--input", type=str, default="benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl")
    p.add_argument("--output", type=str, default="outputs/phase1_truthfulqa_hallu_results_300.jsonl")
    p.add_argument("--manifest", type=str, default="outputs/phase1_run_manifest.json")

    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="bfloat16")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--n_splits", type=int, default=5)

    p.add_argument("--hidden_layers", type=str, default="16") # middle layer

    p.add_argument("--hidden_pooling", type=str, default="mean_answer")
    p.add_argument("--hidden_normalize", action="store_true")

    p.add_argument("--batch_size", type=int, default=8)

    args = p.parse_args()
    layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())

    return Phase1Config(
        input_jsonl=_resolve(args.input),
        output_jsonl=_resolve(args.output),
        output_manifest=_resolve(args.manifest),
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        seed=int(args.seed),
        B=int(args.B),
        ci=float(args.ci),
        n_splits=int(args.n_splits),
        hidden_layers=layers,
        hidden_pooling=str(args.hidden_pooling),
        hidden_normalize=bool(args.hidden_normalize),
        batch_size=int(args.batch_size),
    )


# ----------------------------
# main () starts
# ----------------------------
def main() -> None:
    cfg = parse_args()
    setup_logging(verbosity=1)  # NOTE: Use INFO by default for progress logging.
    set_global_seeds(cfg.seed)

    ensure_parent_dir(cfg.output_jsonl)
    ensure_parent_dir(cfg.output_manifest)

    examples = load_truthfulqa_examples(cfg.input_jsonl)
    if len(examples) == 0:
        die(f"No examples loaded from {cfg.input_jsonl}")

    y = np.asarray([ex.label for ex in examples], dtype=int)
    logger.info("Loaded %d examples (pos=%d, neg=%d)", len(examples), int(np.sum(y == 1)), int(np.sum(y == 0)))

    llm = LLMWrapper(
        model_name=cfg.model_name,
        device=cfg.device,
    )

    # ----------------------------
    # Unsupervised scorers (LNTP, MTP) on frozen answers
    # ----------------------------
    # NOTE: All methods evaluate the same frozen model answers (no generation here).
    questions = [ex.question for ex in examples]
    answers = [ex.model_answer for ex in examples]

    s_lntp_u, s_mtp_u, lntp_stats = compute_lntp_mtp_for_qa_batch(
        llm,
        questions,
        answers,
        orientation="uncertainty",
        return_log_stats=True,
    )

    s_lntp_u_np = np.asarray(s_lntp_u, dtype=np.float64)
    s_mtp_u_np = np.asarray(s_mtp_u, dtype=np.float64)

    logger.info("Done: logit-based scorers (LNTP/MTP).")

    # ----------------------------
    # EGH primitives (Hu et al.): for (a) supervised probe and (b) unsupervised diagnostics
    # ----------------------------
    # NOTE: Compute per-example primitives; later used for OOF probe and for diagnostic AUROCs/plots.
    X_egh_rows: List[List[float]] = []

    # Raw primitives as unsupervised scores (for AUROC + plots)
    s_egh_grad: List[float] = []
    s_egh_emb: List[float] = []
    s_egh_kl: List[float] = []
    s_egh_ce: List[float] = []
    s_egh_entropy: List[float] = []

    k = 0
    sanity_checked = 0

    for i, ex in enumerate(examples):
        q, a = ex.question, ex.model_answer
        prim = compute_egh_primitives_for_qa(llm, q, a, strict=True, chunk_size=None)

        g = float(prim["grad_norm"])
        d = float(prim["d_loss"])
        e = float(prim["emb_diff"])

        # New primitives (Hu-style)
        ce = float(prim["ce_loss"])
        hp = float(prim["h_p"])

        g_vec = prim.get("g_vec", [])
        e_vec = prim.get("e_vec", [])

        # supervised probe uses [G_vector, E_vector] only
        X_egh_rows.append(g_vec + e_vec)

        if len(g_vec) == 0 or len(e_vec) == 0:
            raise RuntimeError(f"Empty g_vec/e_vec at sample {i} (qid={ex.qid}).")


        # unsupervised primitives for reporting/plots
        s_egh_grad.append(g)
        s_egh_emb.append(e)
        s_egh_kl.append(d)
        s_egh_ce.append(ce)
        s_egh_entropy.append(hp)

        # NOTE: Optional spot-check for finiteness / non-degenerate primitives.
        if k > 0 and sanity_checked < k:
            sanity_checked += 1
            if (g == 0.0 and d == 0.0) or (
                not np.isfinite(g)
                or not np.isfinite(d)
                or not np.isfinite(e)
                or not np.isfinite(ce)
                or not np.isfinite(hp)
            ):
                raise RuntimeError(
                    f"Sanity check failed on sample {i} (qid={ex.qid}): "
                    f"grad_norm={g}, d_loss(KL)={d}, emb_diff={e}, ce_loss={ce}, h_p={hp}."
                )

    X_egh = np.asarray(X_egh_rows, dtype=np.float64)

    s_egh_grad_np = np.asarray(s_egh_grad, dtype=np.float64)
    s_egh_emb_np = np.asarray(s_egh_emb, dtype=np.float64)
    s_egh_kl_np = np.asarray(s_egh_kl, dtype=np.float64)
    s_egh_ce_np = np.asarray(s_egh_ce, dtype=np.float64)
    s_egh_entropy_np = np.asarray(s_egh_entropy, dtype=np.float64)

    logger.info("Done: gradient-based primitives (EGH).")

    # ----------------------------
    # Guardrails: scorers should be finite and not constant
    # ----------------------------
    # NOTE: Fail fast if a scorer degenerates or produces NaN/Inf (prevents silent artifacts).
    def _assert_not_constant(name: str, arr: np.ndarray) -> None:
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise RuntimeError(
                f"Non-finite values in scorer: {name} ({n_bad}/{arr.size}). "
                "This often indicates empty/invalid answer spans in LNTP/MTP teacher-forcing scoring. "
                "Fix alignment; do not impute defaults."
            )
        if float(np.nanstd(arr)) == 0.0:
            raise RuntimeError(f"Degenerate scorer (constant) detected: {name}")

    _assert_not_constant("lntp_uncertainty", s_lntp_u_np)
    _assert_not_constant("mtp_uncertainty", s_mtp_u_np)

    _assert_not_constant("egh_grad_norm", s_egh_grad_np)
    _assert_not_constant("egh_emb_diff", s_egh_emb_np)
    _assert_not_constant("egh_kl", s_egh_kl_np)
    _assert_not_constant("egh_ce", s_egh_ce_np)
    _assert_not_constant("egh_entropy", s_egh_entropy_np)

    # ----------------------------
    # Hidden features + OOF (centralized OOF)
    # ----------------------------
    # NOTE: Adapter provides the minimal interface expected by hidden feature extraction.
    @dataclass
    class QAAdapter:
        qid: str
        question: str
        model_answer: str

    qa_like = [QAAdapter(ex.qid, ex.question, ex.model_answer) for ex in examples]

    feature_cfg = HiddenFeatureConfig(
        layers=cfg.hidden_layers,
        pooling=cfg.hidden_pooling,
        normalize=cfg.hidden_normalize,
    )

    # NOTE: Extract hidden-state features once; keep_idx selects usable rows.
    X_hidden, kept_idx, hidden_meta = build_hidden_feature_matrix(qa_like, llm, feature_cfg=feature_cfg)
    kept_idx = np.asarray(kept_idx, dtype=int)

    if kept_idx.size == 0:
        raise RuntimeError("Hidden feature extraction kept 0 samples; cannot train hidden probe.")

    y_hidden = y[kept_idx]

    # NOTE: Out-of-fold probe to avoid leakage (each sample scored by a model not trained on it).
    s_hidden_kept, hidden_folds = oof_logreg_scores(
        X_hidden,
        y_hidden,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
    )
    hidden_auc = compute_auroc(y_hidden, s_hidden_kept)

    logger.info("Done: hidden-state probe (OOF logistic regression).")

    # NOTE: Re-insert hidden probe outputs into full-length array (NaN for dropped rows).
    s_hidden_full = np.full(len(y), np.nan, dtype=np.float64)
    s_hidden_full[kept_idx] = s_hidden_kept

    # EGH probe OOF (same centralized OOF function)
    # NOTE: Probe trained on EGH feature matrix; output is the single reported EGH score.
    s_egh, egh_folds = oof_logreg_scores(
        X_egh,
        y,
        n_splits=cfg.n_splits,
        seed=cfg.seed,
    )
    egh_auc = compute_auroc(y, s_egh)

    # Unsupervised AUROCs
    # NOTE: These quantify the raw score signal without any supervised aggregation.
    lntp_auc = compute_auroc(y, s_lntp_u_np)
    mtp_auc = compute_auroc(y, s_mtp_u_np)

    # Unsupervised AUROCs for Hu-style primitives
    # NOTE: Diagnostics only; useful for ablations/appendix, not necessarily main comparisons.
    egh_grad_auc = compute_auroc(y, s_egh_grad_np)
    egh_emb_auc = compute_auroc(y, s_egh_emb_np)
    egh_kl_auc = compute_auroc(y, s_egh_kl_np)
    egh_ce_auc = compute_auroc(y, s_egh_ce_np)
    egh_entropy_auc = compute_auroc(y, s_egh_entropy_np)

    # ----------------------------
    # Bootstrap confidence intervals
    # ----------------------------
    # NOTE: Stratified bootstrap for stable CI under class imbalance.
    boot_cfg = BootstrapConfig(B=cfg.B, ci=cfg.ci, seed=cfg.seed, stratified=True)

    lntp_boot = bootstrap_auc(y, s_lntp_u_np, boot_cfg)
    mtp_boot = bootstrap_auc(y, s_mtp_u_np, boot_cfg)
    egh_boot = bootstrap_auc(y, s_egh, boot_cfg)
    hidden_boot = bootstrap_auc(y_hidden, s_hidden_kept, boot_cfg)

    egh_grad_boot = bootstrap_auc(y, s_egh_grad_np, boot_cfg)
    egh_emb_boot = bootstrap_auc(y, s_egh_emb_np, boot_cfg)
    egh_kl_boot = bootstrap_auc(y, s_egh_kl_np, boot_cfg)
    egh_ce_boot = bootstrap_auc(y, s_egh_ce_np, boot_cfg)
    egh_entropy_boot = bootstrap_auc(y, s_egh_entropy_np, boot_cfg)

    # NOTE: Delta vs. random baseline (score=0.5 constant) with stored bootstrap indices.
    lntp_delta = bootstrap_auc_diff_with_indices(
        y, s_lntp_u_np, np.full_like(s_lntp_u_np, 0.5), boot_cfg, store_indices=True
    )
    mtp_delta = bootstrap_auc_diff_with_indices(
        y, s_mtp_u_np, np.full_like(s_mtp_u_np, 0.5), boot_cfg, store_indices=True
    )
    egh_delta = bootstrap_auc_diff_with_indices(
        y, s_egh, np.full_like(s_egh, 0.5), boot_cfg, store_indices=True
    )
    hidden_delta = bootstrap_auc_diff_with_indices(
        y_hidden, s_hidden_kept, np.full_like(s_hidden_kept, 0.5), boot_cfg, store_indices=True
    )

    indices_path = cfg.output_manifest.with_suffix(".bootstrap_indices.npz")
    np.savez_compressed(
        indices_path,
        lntp=lntp_delta.indices,
        mtp=mtp_delta.indices,
        egh=egh_delta.indices,
        hidden=hidden_delta.indices,
    )

    # ----------------------------
    # Write per-sample output JSONL
    # ----------------------------
    # NOTE: Per-example outputs support downstream plotting and auditing.
    with cfg.output_jsonl.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            out = {
                "qid": ex.qid,
                "question": ex.question,
                "model_answer": ex.model_answer,
                "hallucinated": int(ex.label),
                "scores": {
                    "lntp_uncertainty": float(s_lntp_u_np[i]),
                    "mtp_uncertainty": float(s_mtp_u_np[i]),
                    "egh_probe_oof": float(s_egh[i]),
                    "egh_grad_norm": float(s_egh_grad_np[i]),
                    "egh_emb_diff": float(s_egh_emb_np[i]),
                    "egh_kl": float(s_egh_kl_np[i]),
                    "egh_ce": float(s_egh_ce_np[i]),
                    "egh_entropy": float(s_egh_entropy_np[i]),
                    "hidden_probe_oof": None if not np.isfinite(s_hidden_full[i]) else float(s_hidden_full[i]),
                },
                "logit_stats": lntp_stats[i],
            }
            f.write(json.dumps(_jsonify(out), ensure_ascii=False) + "\n")

    # ----------------------------
    # Write manifest JSON
    # ----------------------------
    # NOTE: Manifest stores protocol + versions + summary metrics for reproducibility.
    kept_n = int(len(kept_idx))
    total_n = int(len(examples))
    coverage = float(kept_n / total_n) if total_n > 0 else 0.0

    # Should-Fix A: capture runtime versions + key backend flags in manifest
    runtime: Dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "sklearn_version": getattr(sklearn, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device": str(cfg.device),
    }
    if torch.cuda.is_available():
        try:
            runtime["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            runtime["cuda_device_name"] = None

        # SDPA backend flags (defensive: API may differ by torch version)
        cuda_backends = getattr(torch.backends, "cuda", None)
        if cuda_backends is not None:
            for key, fn_name in (
                ("flash_sdp_enabled", "flash_sdp_enabled"),
                ("mem_efficient_sdp_enabled", "mem_efficient_sdp_enabled"),
                ("math_sdp_enabled", "math_sdp_enabled"),
            ):
                fn = getattr(cuda_backends, fn_name, None)
                if callable(fn):
                    try:
                        runtime[key] = bool(fn())
                    except Exception:
                        runtime[key] = None

    manifest: Dict[str, Any] = {
        "phase": "phase1_truthfulqa_hallucination",
        "timestamp": int(time.time()),
        "platform": {
            "python": sys.version,
            "os": platform.platform(),
        },
        "runtime": runtime,  # Should-Fix A (new)
        "config": {
            "input": str(cfg.input_jsonl),
            "output": str(cfg.output_jsonl),
            "manifest": str(cfg.output_manifest),
            "model": cfg.model_name,
            "device": cfg.device,
            "dtype": cfg.torch_dtype,
            "seed": cfg.seed,
            "B": cfg.B,
            "ci": cfg.ci,
            "n_splits": cfg.n_splits,
            "hidden_layers": list(cfg.hidden_layers),
            "hidden_pooling": cfg.hidden_pooling,
            "hidden_normalize": cfg.hidden_normalize,
            "batch_size": cfg.batch_size,
        },
        "dataset": {
            "n": int(len(examples)),
            "n_pos": int(np.sum(y == 1)),
            "n_neg": int(np.sum(y == 0)),
        },
        "scores": {
            "auroc": {
                "lntp_uncertainty": float(lntp_auc),
                "mtp_uncertainty": float(mtp_auc),
                "egh_probe_oof": float(egh_auc),
                "egh_grad_norm": float(egh_grad_auc),
                "egh_emb_diff": float(egh_emb_auc),
                "egh_kl": float(egh_kl_auc),
                "egh_ce": float(egh_ce_auc),
                "egh_entropy": float(egh_entropy_auc),
                "hidden_probe_oof": float(hidden_auc),
            },
            "bootstrap": {
                "lntp_uncertainty": bootstrap_result_to_dict(lntp_boot),
                "mtp_uncertainty": bootstrap_result_to_dict(mtp_boot),
                "egh_probe_oof": bootstrap_result_to_dict(egh_boot),
                "egh_grad_norm": bootstrap_result_to_dict(egh_grad_boot),
                "egh_emb_diff": bootstrap_result_to_dict(egh_emb_boot),
                "egh_kl": bootstrap_result_to_dict(egh_kl_boot),
                "egh_ce": bootstrap_result_to_dict(egh_ce_boot),
                "egh_entropy": bootstrap_result_to_dict(egh_entropy_boot),
                "hidden_probe_oof": bootstrap_result_to_dict(hidden_boot),
            },
            "delta_vs_random": {
                "lntp": bootstrap_result_to_dict(lntp_delta),
                "mtp": bootstrap_result_to_dict(mtp_delta),
                "egh": bootstrap_result_to_dict(egh_delta),
                "hidden": bootstrap_result_to_dict(hidden_delta),
                "bootstrap_indices_path": str(indices_path),
            },
        },
        "hidden_probe": {
            "kept_n": kept_n,
            "dropped_n": int(len(examples) - kept_n),
            "coverage": coverage,
            "kept_idx": kept_idx.tolist(),
            "folds": hidden_folds,
            "meta": hidden_meta,
        },
        "egh_probe": {
            "folds": egh_folds,
        },
    }

    with cfg.output_manifest.open("w", encoding="utf-8") as f:
        f.write(json.dumps(_jsonify(manifest), ensure_ascii=False, indent=2) + "\n")

    logger.info("Done. Wrote results to %s and manifest to %s", cfg.output_jsonl, cfg.output_manifest)


if __name__ == "__main__":
    main()  # NOTE: CLI entry point.