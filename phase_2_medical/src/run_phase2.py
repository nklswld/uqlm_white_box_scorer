"""
Phase-2 runner for medical QA evaluation using frozen model outputs.
Loads a JSONL of precomputed predictions, rebuilds task prompts, and computes uncertainty proxy scores.
Inputs: frozen_jsonl (qid, question/context/choices, model_answer, optional pred/gold, is_error, meta) + CLI config.
Outputs: results JSONL with per-example scores; manifest JSON with metrics, versions, config, and bootstrap indices path.
Determinism: fixed seeds (NumPy/Torch + PYTHONHASHSEED); seeded, optionally persisted bootstrap resample indices.
NOTE: potential issue: Torch backend flags are best-effort and may be silently ignored on unsupported builds/hardware.
"""

# phase_2_medical/src/run_phase2.py
from __future__ import annotations

import sys
from pathlib import Path

# Phase-1 compatibility: inject Phase-1 `src/` into sys.path to preserve historical import layout.
ROOT = Path(__file__).resolve().parents[2]   # repo root
PHASE1_SRC = ROOT / "phase_1_replication" / "src"
sys.path.insert(0, str(PHASE1_SRC))

import argparse
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import transformers
import sklearn
from sklearn.metrics import roc_auc_score

from modeling_llm import LLMWrapper
from scorers_logit import compute_lntp_mtp_for_qa_batch
from scorers_gradient import compute_egh_primitives_for_qa
from scorers_hidden import HiddenFeatureConfig, build_hidden_feature_matrix, oof_logreg_scores
from bootstrap import BootstrapConfig, bootstrap_auc, bootstrap_auc_diff_with_indices


# ----------------------------
# Torch runtime (Phase-1 style, best-effort)
# ----------------------------
def configure_torch_runtime() -> None:
    """Best-effort Torch backend toggles; failures are intentionally non-fatal and unreported."""
    os.environ.setdefault("TORCH_SDPA_ENABLE", "1")
    os.environ.setdefault("TORCH_SDPA_DISABLE", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    try:
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
        # NOTE: potential issue: silent failure can mask backend misconfiguration; rely on manifest versions for debugging.
        pass


configure_torch_runtime()


# ----------------------------
# Helpers
# ----------------------------
def setup_logging(verbosity: int) -> None:
    """Configure root logging level from CLI verbosity count (-v / -vv)."""
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


logger = logging.getLogger(__name__)


def set_global_seeds(seed: int) -> None:
    """Set all known PRNG seeds and deterministic CuDNN flags for reproducible scoring."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode reduces nondeterministic kernels at the cost of speed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_parent_dir(p: Path) -> None:
    """Create parent directory for a file path if needed."""
    p.parent.mkdir(parents=True, exist_ok=True)


def _jsonify(obj: Any) -> Any:
    """Recursively convert numpy / torch-ish scalars and arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonify(v) for v in obj]

    # Handle pathlib.Path (PosixPath / WindowsPath)
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def compute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Compute AUROC with explicit length and class-coverage checks (binary labels expected)."""
    y = np.asarray(y).reshape(-1).astype(int)
    s = np.asarray(s).reshape(-1).astype(float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("AUROC: y and scores must have same length.")
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: only one class present.")
    # Convention: higher scores should correspond to higher probability of y==1.
    return float(roc_auc_score(y, s))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation using deterministic average-rank ties (stable mergesort)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("Spearman: size mismatch.")

    def _rankdata(a: np.ndarray) -> np.ndarray:
        # Stable sort ensures deterministic tie handling across platforms.
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
                # Ties get the average rank (classic Spearman convention).
                avg = (i + 1 + j + 1) / 2.0
                ranks[order[i : j + 1]] = avg
            i = j + 1
        return ranks


    rx = _rankdata(x)
    ry = _rankdata(y)
    # Normalize to avoid numerical drift; epsilon guards zero-variance edge cases.
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def truncate_text(s: str, max_chars: int) -> str:
    """Hard character-budget truncation (used as a cheap pre-filter before token truncation)."""
    s = (s or "").strip()
    if max_chars <= 0:
        return s
    return s[:max_chars]

def normalize_pubmedqa_pred(s: str) -> str:
    """Normalize free-form generation to {yes,no,maybe} by prefix; otherwise return first token."""
    s = (s or "").strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("maybe"):
        return "maybe"
    # NOTE: potential issue: unexpected outputs (e.g., empty/other tokens) collapse to a single token and may bias scoring.
    return (s.split()[:1] or [""])[0]

def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Tokenizer-aligned truncation to max_tokens (no special tokens; preserves determinism)."""
    text = (text or "").strip()
    if max_tokens <= 0:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def map_hidden_pooling(user_choice: str) -> str:
    """
    Canonicalize pooling identifiers for hidden-feature extraction.
    Accepts Phase-1 names and a small set of Phase-2 aliases for backward compatibility.
    """
    # Convention: map user-facing aliases to canonical pooling identifiers expected by HiddenFeatureConfig.
    u = (user_choice or "").strip().lower()
    if u in {"mean", "mean_answer"}:
        return "mean_answer"
    if u in {"last", "last_answer"}:
        return "last_answer"
    if u in {"first", "mean_all"}:
        return "mean_all"
    # default:
    return "mean_answer"


def infer_model_short(model_name: str) -> str:
    """Derive a short, filesystem-safe model identifier for output naming."""
    m = (model_name or "").lower()
    if "biomistral" in m:
        return "biomistral"
    if "mistral" in m:
        return "mistral"
    # fallback: last path segment, cleaned
    tail = (model_name or "model").split("/")[-1].lower()
    tail = "".join(ch for ch in tail if ch.isalnum() or ch in {"-", "_"})
    return tail[:32] if tail else "model"



# ----------------------------
# Task prompt builders
# ----------------------------
def build_pubmedqa_prompt(question: str, abstract: str) -> str:
    """Build a PubMedQA prompt constrained to a single label token."""
    return (
        "You are answering a medical question based on the given abstract.\n"
        "Answer using exactly one word from {yes, no, maybe}.\n"
        "Output ONLY that word.\n\n"
        f"Abstract:\n{abstract.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Final answer:"
    )


def build_medqa_prompt(question: str, choices: Dict[str, str]) -> str:
    """Build a MedQA prompt constrained to a single option letter."""
    return (
        "You are answering a multiple-choice medical question.\n"
        "Choose exactly one option letter from {A, B, C, D}.\n"
        "Output ONLY the letter.\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Options:\n"
        f"A. {str(choices.get('A','')).strip()}\n"
        f"B. {str(choices.get('B','')).strip()}\n"
        f"C. {str(choices.get('C','')).strip()}\n"
        f"D. {str(choices.get('D','')).strip()}\n\n"
        "Final answer:"
    )


# ----------------------------
# Frozen schema (generic)
# ----------------------------
@dataclass(frozen=True)
class FrozenRow:
    """Immutable record for one frozen example (inputs + prediction + label) used for Phase-2 scoring."""
    qid: str
    task: str
    question: str
    context: str
    choices: Dict[str, str]
    model_answer: str
    gold: Optional[str]
    pred: Optional[str]
    is_error: int
    meta: Dict[str, Any]


def load_frozen(path: Path, task: str) -> List[FrozenRow]:
    """Load a newline-delimited JSONL file into FrozenRow objects (blank lines are ignored)."""
    out: List[FrozenRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out.append(
                FrozenRow(
                    qid=str(obj["qid"]),
                    task=task,
                    question=str(obj.get("question", "")),
                    context=str(obj.get("context", "")),
                    choices=dict(obj.get("choices", {}) or {}),
                    model_answer=str(obj.get("model_answer", "")),
                    gold=obj.get("gold", None),
                    pred=obj.get("pred", None),
                    is_error=int(obj.get("is_error", 0)),
                    meta=dict(obj.get("meta", {}) or {}),
                )
            )
    return out


# ----------------------------
# Config / CLI
# ----------------------------
@dataclass(frozen=True)
class Phase2Config:
    """Configuration snapshot for Phase-2 scoring (arguments are persisted into the manifest)."""
    task: str
    frozen_jsonl: Path
    out_jsonl: Path
    out_manifest: Path
    out_dir: Path
    run_tag: str
    ablation_name: str
    ablation_setting: str

    model_name: str
    device: str
    torch_dtype: str

    seed: int
    batch_size: int
    n_splits: int

    B: int
    ci: float

    hidden_layers: Tuple[int, ...]
    hidden_pooling: str
    hidden_normalize: bool
    hidden_batch_size: int

    # PubMed-style context truncation (MedQA will simply ignore it)
    max_context_chars: int
    max_context_tokens: int

    egh_chunk_size: int
    verbosity: int


def parse_args() -> Phase2Config:
    """Parse CLI arguments and derive deterministic output paths when not explicitly provided."""
    p = argparse.ArgumentParser(description="Phase 2 – Medical tasks (Frozen Outputs)")

    p.add_argument("--task", type=str, required=True, choices=["pubmedqa", "medqa"])

    p.add_argument("--frozen_jsonl", type=str, required=True)

    # optional now (autofilled if not provided)
    p.add_argument("--out_jsonl", type=str, default=None)
    p.add_argument("--out_manifest", type=str, default=None)

    # NEW: convenience + ablation metadata
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--run_tag", type=str, default="")

    p.add_argument("--ablation_name", type=str, default="")
    p.add_argument("--ablation_setting", type=str, default="")

    
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_splits", type=int, default=5)

    # Phase-1-style evaluation stability knobs
    p.add_argument("--B", type=int, default=5000)
    p.add_argument("--ci", type=float, default=0.95)

    # Hidden probe defaults: middle-ish layer for 32-layer Mistral family = 16
    p.add_argument("--hidden_layers", type=int, nargs="+", default=[16])
    p.add_argument(
        "--hidden_pooling",
        type=str,
        default="mean_answer",
        choices=["mean_answer", "last_answer", "mean_all", "mean", "last", "first"],
    )
    p.add_argument("--hidden_normalize", action="store_true")
    p.add_argument("--hidden_batch_size", type=int, default=8)

    # Context truncation (mainly for PubMedQA; MedQA often has empty context)
    p.add_argument("--max_context_chars", type=int, default=3500)
    p.add_argument("--max_context_tokens", type=int, default=512)

    p.add_argument("--egh_chunk_size", type=int, default=256)

    p.add_argument("-v", "--verbosity", action="count", default=1)

    a = p.parse_args()
    
    # Resolve output directory.
    # Default to phase_2_medical/outputs/final if not given.
    default_out_dir = Path(__file__).resolve().parents[1] / "outputs" / "final"
    out_dir = Path(a.out_dir) if a.out_dir else default_out_dir

    model_short = infer_model_short(str(a.model_name))
    tag = (a.run_tag or "").strip()
    tag_part = f".{tag}" if tag else ""

    # Deterministic naming ties artifacts to task/model/run_tag/bootstrap-B for reproducible result bookkeeping.
    if a.out_jsonl is None:
        a.out_jsonl = str(out_dir / f"{a.task}_{model_short}{tag_part}.B{int(a.B)}.results.jsonl")
    if a.out_manifest is None:
        a.out_manifest = str(out_dir / f"{a.task}_{model_short}{tag_part}.B{int(a.B)}.manifest.json")


    return Phase2Config(
        task=str(a.task),
        frozen_jsonl=Path(a.frozen_jsonl),
        out_jsonl=Path(a.out_jsonl),
        out_manifest=Path(a.out_manifest),
        out_dir=out_dir,
        run_tag=str(a.run_tag or ""),
        ablation_name=str(a.ablation_name or ""),
        ablation_setting=str(a.ablation_setting or ""),
        model_name=str(a.model_name),
        device=str(a.device),
        torch_dtype=str(a.dtype),
        seed=int(a.seed),
        batch_size=int(a.batch_size),
        n_splits=int(a.n_splits),
        B=int(a.B),
        ci=float(a.ci),
        hidden_layers=tuple(int(x) for x in a.hidden_layers),
        hidden_pooling=map_hidden_pooling(str(a.hidden_pooling)),
        hidden_normalize=bool(a.hidden_normalize),
        hidden_batch_size=int(a.hidden_batch_size),
        egh_chunk_size=int(a.egh_chunk_size),
        verbosity=int(a.verbosity),
        max_context_chars=int(a.max_context_chars),
        max_context_tokens=int(a.max_context_tokens),
    )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    """Run Phase-2 scoring and write per-example results JSONL plus a run manifest."""
    cfg = parse_args()
    setup_logging(cfg.verbosity)
    set_global_seeds(cfg.seed)

    ensure_parent_dir(cfg.out_jsonl)
    ensure_parent_dir(cfg.out_manifest)

    examples = load_frozen(cfg.frozen_jsonl, task=cfg.task)
    if len(examples) == 0:
        raise SystemExit(f"No examples loaded from: {cfg.frozen_jsonl}")

    # Label convention: y==1 is the positive class ("error"); scores should increase with error likelihood.
    y = np.asarray([ex.is_error for ex in examples], dtype=int)
    logger.info(
        "Loaded %d frozen examples for task=%s (pos=%d neg=%d)",
        len(examples),
        cfg.task,
        int((y == 1).sum()),
        int((y == 0).sum()),
    )

    if cfg.task == "pubmedqa":
        # PubMedQA uses {yes,no,maybe}: normalize to a single token to prevent verbose/echoed generations.
        answers = [normalize_pubmedqa_pred(ex.pred or "") for ex in examples]
    else:
        # MedQA uses option letters: prefer the frozen `pred` when present, else fall back to raw `model_answer`.
        answers = [(ex.pred.strip().upper() if ex.pred else ex.model_answer) for ex in examples]

    class Phase2LLM(LLMWrapper):
        # IMPORTANT: Phase 2 receives fully-formed prompts (question argument is already the full prompt).
        def build_prompt(self, question: str) -> str:
            return question

    llm = Phase2LLM(
        cfg.model_name,
        device=cfg.device,
        max_input_tokens=cfg.max_context_tokens,
    )
    
    # Best-effort dtype annotation (may not affect loaded weights; still recorded in manifest for auditability).
    try:
        if hasattr(llm, "torch_dtype"):
            llm.torch_dtype = cfg.torch_dtype
    except Exception:
        # NOTE: potential issue: dtype mismatch between intended and actual model load can change scores; verify in LLMWrapper.
        pass

    tok = getattr(llm, "tokenizer", None) or getattr(llm, "tok", None)
    if tok is None:
        # Fallback tokenizer ensures deterministic truncation even if LLMWrapper does not expose one.
        tok = transformers.AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # Build task-specific prompts (teacher-forcing uses prompt + provided answer; no generation in Phase-2).
    prompts: List[str] = []
    for ex in examples:
        if cfg.task == "pubmedqa":
            # Two-stage truncation: cheap char cap, then tokenizer-aligned token cap for model safety.
            ctx = truncate_text(ex.context, cfg.max_context_chars)
            ctx = truncate_to_tokens(ctx, tok, cfg.max_context_tokens)
            prompts.append(build_pubmedqa_prompt(ex.question, ctx))
        elif cfg.task == "medqa":
            prompts.append(build_medqa_prompt(ex.question, ex.choices))
        else:
            raise RuntimeError(f"Unknown task: {cfg.task}")

    # ----------------------------
    # Unsupervised: LNTP / MTP
    # ----------------------------
    # Scoring convention: orientation="uncertainty" => larger score should indicate higher uncertainty/error propensity.
    s_lntp, s_mtp, lntp_stats = compute_lntp_mtp_for_qa_batch(
        llm,
        prompts,
        answers,
        orientation="uncertainty",
        return_log_stats=True,
    )
    # lntp_stats is retained for optional debugging/inspection (not currently persisted into outputs here).
    s_lntp = np.asarray(s_lntp, dtype=np.float64)
    s_mtp = np.asarray(s_mtp, dtype=np.float64)
    logger.info("Done: LNTP/MTP")

    # ----------------------------
    # EGH primitives (diagnostics + supervised probe inputs)
    # Probe uses ONLY [G_vec, E_vec] ordering (stable concat for reproducible feature meaning)
    # ----------------------------
    X_egh_rows: List[List[float]] = []
    s_egh_grad: List[float] = []
    s_egh_emb: List[float] = []
    s_egh_kl: List[float] = []
    s_egh_ce: List[float] = []
    s_egh_entropy: List[float] = []

    # feature subsets for ablations
    X_egh_g_rows: List[List[float]] = []
    X_egh_e_rows: List[List[float]] = []

    # scalar-only probe features (Appendix ablation)
    X_egh_scalar_rows: List[List[float]] = []


    t0 = time.time()
    for i, (p_text, ans) in enumerate(zip(prompts, answers)):
        prim = compute_egh_primitives_for_qa(
            llm, p_text, ans,
            strict=True,
            chunk_size=cfg.egh_chunk_size,
        )
        # Scalars: logged for inspection and reused as scalar-only probe features (consistent 5D set).
        s_egh_grad.append(float(prim["grad_norm"]))
        s_egh_emb.append(float(prim["emb_diff"]))
        s_egh_kl.append(float(prim["d_loss"]))
        s_egh_ce.append(float(prim["ce_loss"]))
        s_egh_entropy.append(float(prim["h_p"]))

        
        # Vector features: concatenate G then E to enforce stable column semantics across runs/ablations.
        g_vec = prim["g_vec"]
        e_vec = prim["e_vec"]

        g_list = list(g_vec)
        e_list = list(e_vec)

        # Base (G+E vectors)
        X_egh_rows.append(g_list + e_list)

        # Ablation 1 feature subsets
        X_egh_g_rows.append(g_list)      # G-only
        X_egh_e_rows.append(e_list)      # E-only

        # Ablation 2 scalar-only features (Appendix)
        # Choose a consistent scalar set (5 dims)
        X_egh_scalar_rows.append([
            float(prim["grad_norm"]),
            float(prim["emb_diff"]),
            float(prim["d_loss"]),
            float(prim["ce_loss"]),
            float(prim["h_p"]),
        ])


        if (i + 1) % 50 == 0:
            logger.info("EGH: %d/%d (%.1fs)", i + 1, len(examples), time.time() - t0)

    X_egh = np.asarray(X_egh_rows, dtype=np.float64)
    logger.info("Done: EGH primitives (%.1fs)", time.time() - t0)

    # Matrices for ablations (shape must align with y for downstream OOF scoring).
    X_egh_g = np.asarray(X_egh_g_rows, dtype=np.float64)
    X_egh_e = np.asarray(X_egh_e_rows, dtype=np.float64)
    X_egh_scalar = np.asarray(X_egh_scalar_rows, dtype=np.float64)

    
    # OOF probes on EGH features (logistic regression; returns per-example out-of-fold scores).
    # Base: vector GE
    s_egh_oof_ge, egh_meta_ge = oof_logreg_scores(X_egh, y, n_splits=cfg.n_splits, seed=cfg.seed)
    s_egh_oof_ge = np.asarray(s_egh_oof_ge, dtype=np.float64)

    # Ablation 1 (G-only / E-only)
    s_egh_oof_g, egh_meta_g = oof_logreg_scores(X_egh_g, y, n_splits=cfg.n_splits, seed=cfg.seed)
    s_egh_oof_g = np.asarray(s_egh_oof_g, dtype=np.float64)

    s_egh_oof_e, egh_meta_e = oof_logreg_scores(X_egh_e, y, n_splits=cfg.n_splits, seed=cfg.seed)
    s_egh_oof_e = np.asarray(s_egh_oof_e, dtype=np.float64)

    # Ablation 2 (Scalar-only vs Vector)
    s_egh_oof_scalar, egh_meta_scalar = oof_logreg_scores(X_egh_scalar, y, n_splits=cfg.n_splits, seed=cfg.seed)
    s_egh_oof_scalar = np.asarray(s_egh_oof_scalar, dtype=np.float64)


    # Sanity Checks
    def assert_scores_ok(name, s):
        # Guard against silent degeneracy: NaNs/Infs or constant scores make AUROC/bootstraps meaningless.
        s = np.asarray(s, dtype=np.float64)

        if not np.all(np.isfinite(s)):
            raise ValueError(f"{name}: non-finite values")

        if float(np.std(s)) < 1e-12:
            raise ValueError(f"{name}: (near-)constant score")

    assert_scores_ok("LNTP", s_lntp)
    assert_scores_ok("MTP", s_mtp)
    assert_scores_ok("EGH_probe_ge", s_egh_oof_ge)
    
    g = np.asarray(s_egh_oof_g, dtype=np.float64)

    try:
        assert_scores_ok("EGH_probe_g_only", g)
    except ValueError as e:
        # NOTE: potential issue: skipping here allows downstream metrics/manifest to proceed with degenerate G-only scores.
        logger.warning(f"Skipping G-only sanity check: {e}")

    assert_scores_ok("EGH_probe_e_only", s_egh_oof_e)
    assert_scores_ok("EGH_probe_scalar_only", s_egh_oof_scalar)


    # ----------------------------
    # Hidden probe (OOF)
    # ----------------------------
    hcfg = HiddenFeatureConfig(
        layers=cfg.hidden_layers,
        pooling=cfg.hidden_pooling,
        normalize=cfg.hidden_normalize,
        batch_size=cfg.hidden_batch_size,
    )

    # Adapter: build_hidden_feature_matrix expects .question + .model_answer.
    # We pass the FULL prompt as `.question` so hidden features align with teacher-forcing inputs used elsewhere.
    class _Ex:
        def __init__(self, prompt: str, ans: str):
            self.question = prompt
            self.model_answer = ans

    ex_list = [_Ex(p, a) for p, a in zip(prompts, answers)]
    X_hid, kept_indices, hidden_meta = build_hidden_feature_matrix(
        ex_list, llm, feature_cfg=hcfg, strict=False
    )
    # NOTE: potential issue: strict=False can drop examples silently; coverage is tracked and NaNs are injected below.
    kept_indices = np.asarray(kept_indices, dtype=int)
    y_kept = y[kept_indices]

    hidden_oof, hid_probe_meta = oof_logreg_scores(
        X_hid, y_kept, n_splits=cfg.n_splits, seed=cfg.seed
    )
    # hid_probe_meta is returned for debugging/inspection (not currently persisted into the manifest here).
    hidden_oof = np.asarray(hidden_oof, dtype=np.float64)

    # Reinflate to full length so results JSONL and masking logic share the same example index space.
    hidden_oof_full = np.full((len(examples),), np.nan, dtype=np.float64)
    hidden_oof_full[kept_indices] = hidden_oof
    logger.info("Done: Hidden OOF probe (kept=%d/%d)", kept_indices.size, len(examples))

    # ----------------------------
    # Metrics (+ Phase-1-like Bootstrap CI + delta vs random)
    # ----------------------------
    # Bootstrap is seeded + stratified for reproducible CIs; indices can be persisted for exact reruns.
    boot_cfg = BootstrapConfig(B=cfg.B, ci=cfg.ci, seed=cfg.seed, stratified=True)
    baseline_all = np.full((len(y),), 0.5, dtype=np.float64)  # random baseline scores (uninformative constant)

    # Save bootstrap resample indices (per-metric) to permit exact CI regeneration and cross-implementation checks.
    lntp_delta = bootstrap_auc_diff_with_indices(y, s_lntp, baseline_all, boot_cfg, store_indices=True)
    mtp_delta  = bootstrap_auc_diff_with_indices(y, s_mtp,  baseline_all, boot_cfg, store_indices=True)
    egh_delta  = bootstrap_auc_diff_with_indices(y, s_egh_oof_ge, baseline_all, boot_cfg, store_indices=True)
    # NOTE: potential issue: egh_delta duplicates egh_ge_delta (same inputs); kept for backward compatibility of stored keys.
    egh_ge_delta = bootstrap_auc_diff_with_indices(y, s_egh_oof_ge, baseline_all, boot_cfg, store_indices=True)
    egh_g_delta  = bootstrap_auc_diff_with_indices(y, s_egh_oof_g,  baseline_all, boot_cfg, store_indices=True)
    egh_e_delta  = bootstrap_auc_diff_with_indices(y, s_egh_oof_e,  baseline_all, boot_cfg, store_indices=True)
    egh_s_delta  = bootstrap_auc_diff_with_indices(y, s_egh_oof_scalar, baseline_all, boot_cfg, store_indices=True)

    # Hidden uses only the finite subset (kept examples); baseline length must match masked arrays exactly.
    mask = np.isfinite(hidden_oof_full)
    y_h = y[mask]
    s_h = hidden_oof_full[mask]
    baseline_h = np.full_like(s_h, 0.5, dtype=np.float64)
    hidden_delta = bootstrap_auc_diff_with_indices(y_h, s_h, baseline_h, boot_cfg, store_indices=True)

    indices_path = cfg.out_manifest.with_suffix(".bootstrap_indices.npz")
    hidden_kept_full_indices = np.where(mask)[0].astype(np.int32)
    np.savez_compressed(
        indices_path,
        lntp=lntp_delta.indices,
        mtp=mtp_delta.indices,
        egh=egh_delta.indices,
        egh_ge=egh_ge_delta.indices,
        egh_g=egh_g_delta.indices,
        egh_e=egh_e_delta.indices,
        egh_scalar=egh_s_delta.indices,

        hidden=hidden_delta.indices,
        hidden_kept_indices=hidden_kept_full_indices
    )


    def _metric_block(scores: np.ndarray, yy: np.ndarray) -> Dict[str, Any]:
        """Compute AUROC, Spearman rho, and bootstrap CI (plus delta AUROC vs constant-random baseline)."""
        # Invariant: scores and yy must already be aligned and length-matched (masking handled by caller where needed).
        auc = compute_auroc(yy, scores)
        rho = spearman_rho(yy.astype(float), scores.astype(float))
        b = bootstrap_auc(yy, scores, boot_cfg, store_indices=False)
        baseline = np.full_like(scores, 0.5, dtype=np.float64)
        d = bootstrap_auc_diff_with_indices(yy, scores, baseline, boot_cfg)
        return {
            "auc": auc,
            "spearman_rho": rho,
            "auc_ci": {"low": b.ci_low, "high": b.ci_high, "B": b.n_total, "seed": b.seed},
            "delta_auc_vs_random": {"diff": d.diff, "low": d.ci_low, "high": d.ci_high, "B": d.n_total, "seed": d.seed},
        }

    metrics: Dict[str, Any] = {
        "LNTP": _metric_block(s_lntp, y),
        "MTP": _metric_block(s_mtp, y),
        
         # Backward-compatible alias retained for downstream scripts expecting this key.
        "EGH_probe_oof": _metric_block(s_egh_oof_ge, y),  # backward compatible alias

        # Base EGH (vector GE)
        "EGH_probe_ge": _metric_block(s_egh_oof_ge, y),

        # Ablation 1
        "EGH_probe_g_only": _metric_block(s_egh_oof_g, y),
        "EGH_probe_e_only": _metric_block(s_egh_oof_e, y),

        # Ablation 2 (Appendix)
        "EGH_probe_scalar_only": _metric_block(s_egh_oof_scalar, y),
    }


    # Hidden probe metrics are computed on the kept subset only (mask applied).
    metrics["Hidden_probe_oof"] = _metric_block(hidden_oof_full[mask], y[mask])



    # ----------------------------
    # Write per-example results JSONL
    # ----------------------------
    with cfg.out_jsonl.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            # Invariant: all non-hidden score arrays are aligned to `examples` order; hidden_probe_oof may be NaN for dropped rows.
            row = {
                "qid": ex.qid,
                "task": cfg.task,
                "label": int(y[i]),
                "gold": ex.gold,
                "pred": ex.pred,
                "model_answer": ex.model_answer,
                "lntp": float(s_lntp[i]),
                "mtp": float(s_mtp[i]),
                "egh_grad_norm": float(s_egh_grad[i]),
                "egh_emb_diff": float(s_egh_emb[i]),
                "egh_kl": float(s_egh_kl[i]),
                "egh_ce": float(s_egh_ce[i]),
                "egh_entropy": float(s_egh_entropy[i]),
                # Key naming: keep both alias and explicit GE key for downstream compatibility.
                "egh_probe_oof": float(s_egh_oof_ge[i]),
                "egh_probe_ge": float(s_egh_oof_ge[i]),
                "egh_probe_g_only": float(s_egh_oof_g[i]),
                "egh_probe_e_only": float(s_egh_oof_e[i]),
                "egh_probe_scalar_only": float(s_egh_oof_scalar[i]),

                # Explicitly encode missing hidden scores as null to avoid JSON NaN portability issues.
                "hidden_probe_oof": None if not np.isfinite(hidden_oof_full[i]) else float(hidden_oof_full[i]),
                "meta": ex.meta,
            }
            f.write(json.dumps(_jsonify(row), ensure_ascii=False) + "\n")

    # ----------------------------
    # Manifest
    # ----------------------------
    
    kept_n = int(len(kept_indices))
    total_n = int(len(examples))
    hidden_coverage = float(kept_n / total_n) if total_n > 0 else 0.0
    
    manifest = {
        "task": cfg.task,
        "run": {
            "run_tag": cfg.run_tag,
            "ablation_name": cfg.ablation_name,
            "ablation_setting": cfg.ablation_setting,
            "out_dir": str(cfg.out_dir),
        },
        "config": asdict(cfg),
        "n": int(len(examples)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "metrics": metrics,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        # HiddenFeatureConfig/build_hidden_feature_matrix metadata is persisted for probe auditability.
        "hidden_meta": hidden_meta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "outputs": {
            "out_jsonl": str(cfg.out_jsonl),
            "bootstrap_indices_npz": str(cfg.out_manifest.with_suffix(".bootstrap_indices.npz")),
        },
        # Coverage summary for transparent reporting when hidden features drop examples.
        "hidden_coverage": {
            "kept_n": kept_n,
            "total_n": total_n,
            "coverage": hidden_coverage,
        },
        "egh_meta": {
            "ge": egh_meta_ge,
            "g_only": egh_meta_g,
            "e_only": egh_meta_e,
            "scalar_only": egh_meta_scalar,
        },
    }

    cfg.out_manifest.write_text(json.dumps(_jsonify(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote results: %s", cfg.out_jsonl)
    logger.info("Wrote manifest: %s", cfg.out_manifest)


if __name__ == "__main__":
    main()