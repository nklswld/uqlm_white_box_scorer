from __future__ import annotations
import sys, json, time, platform
from pathlib import Path

import numpy as np
import torch
import transformers
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]  # phase_2_medical
PHASE1_SRC = ROOT.parent / "phase_1_replication" / "src"
sys.path.insert(0, str(PHASE1_SRC))

from modeling_llm import LLMWrapper
from scorers_logit import compute_lntp_mtp_for_qa_batch


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    order_x = np.argsort(x, kind="mergesort")
    order_y = np.argsort(y, kind="mergesort")
    rx = np.empty_like(order_x, dtype=float)
    ry = np.empty_like(order_y, dtype=float)
    rx[order_x] = np.arange(1, x.size + 1, dtype=float)
    ry[order_y] = np.arange(1, y.size + 1, dtype=float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def compute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).reshape(-1).astype(int)
    s = np.asarray(s).reshape(-1).astype(float)
    if np.unique(y).size < 2:
        raise ValueError("AUROC undefined: only one class present.")
    return float(roc_auc_score(y, s))


def load_frozen_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_model_key(model_name: str) -> str:
    mn = (model_name or "").lower()
    if "biomistral" in mn:
        return "biomistral"
    # default bucket
    return "mistral"


def main():
    import argparse

    p = argparse.ArgumentParser()

    # Convenience mode: if you run the script without args, use defaults.
    p.add_argument(
        "--run_defaults",
        action="store_true",
        help="Run with built-in default paths/settings (no need to pass CLI args).",
    )

    # Required (auto-filled via --run_defaults)
    p.add_argument("--frozen_jsonl", default=None)
    p.add_argument("--out_jsonl", default=None)
    p.add_argument("--out_manifest", default=None)

    # Optional separate k-sweep manifest path
    p.add_argument("--out_k_sweep_manifest", type=str, default=None)

    # Model / runtime
    p.add_argument("--model_name", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_input_tokens", type=int, default=512)

    # Answer-Span Ablation (k-sweep)
    p.add_argument(
        "--k_list",
        type=int,
        nargs="+",
        default=None,
        help="If set: run Answer-Span k-sweep using first k answer tokens.",
    )

    a = p.parse_args()

    # -----------------------
    # Built-in defaults (MATCH YOUR VM STRUCTURE)
    # -----------------------
    DEFAULT_FROZEN = ROOT / "outputs" / "frozen" / "truthfulqa_hallu_mistral_like.jsonl"
    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    DEFAULT_K_LIST = [1, 2, 3, 5, 10, 20]

    if a.run_defaults:
        if a.model_name is None:
            a.model_name = DEFAULT_MODEL
        if a.k_list is None:
            a.k_list = DEFAULT_K_LIST
        if a.frozen_jsonl is None:
            a.frozen_jsonl = str(DEFAULT_FROZEN)

        # Output folder follows your existing convention:
        # phase_2_medical/outputs/ablations/token_score_bias/<model_key>/
        model_key = infer_model_key(a.model_name)
        default_outdir = ROOT / "outputs" / "ablations" / "token_score_bias" / model_key

        if a.out_jsonl is None:
            a.out_jsonl = str(default_outdir / "results.jsonl")
        if a.out_manifest is None:
            a.out_manifest = str(default_outdir / "manifest.json")
        if a.out_k_sweep_manifest is None:
            a.out_k_sweep_manifest = str(default_outdir / "k_sweep.manifest.json")

    # If still missing required args, error clearly
    missing = []
    if a.frozen_jsonl is None:
        missing.append("--frozen_jsonl")
    if a.out_jsonl is None:
        missing.append("--out_jsonl")
    if a.out_manifest is None:
        missing.append("--out_manifest")
    if a.model_name is None:
        missing.append("--model_name")
    if missing:
        raise SystemExit(
            "Missing required args: "
            + ", ".join(missing)
            + "\nEither pass them explicitly or run with --run_defaults."
        )

    # -----------------------
    # Debug prints
    # -----------------------
    print("[INFO] frozen_jsonl:", a.frozen_jsonl)
    print("[INFO] out_jsonl:", a.out_jsonl)
    print("[INFO] out_manifest:", a.out_manifest)
    print("[INFO] out_k_sweep_manifest:", a.out_k_sweep_manifest)
    print("[INFO] model_name:", a.model_name)
    print("[INFO] device:", a.device, "| dtype:", a.dtype, "| seed:", a.seed)
    print("[INFO] batch_size:", a.batch_size, "| max_input_tokens:", a.max_input_tokens)
    print("[INFO] k_list:", a.k_list)

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    frozen_path = Path(a.frozen_jsonl)

    # Explicit sanity check with helpful listing
    if not frozen_path.exists():
        frozen_dir = ROOT / "outputs" / "frozen"
        available = []
        if frozen_dir.exists():
            available = sorted([p.name for p in frozen_dir.glob("*.jsonl")])
        msg = (
            f"[ERROR] frozen_jsonl not found: {frozen_path}\n"
            f"[HINT] Available in {frozen_dir}:\n  - " + "\n  - ".join(available)
        )
        raise SystemExit(msg)

    out_jsonl = Path(a.out_jsonl)
    out_manifest = Path(a.out_manifest)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = load_frozen_jsonl(frozen_path)
    if not rows:
        raise SystemExit(f"[ERROR] No rows: {frozen_path}")

    # Labels: truthfulqa frozen may not use is_error -> but your sample uses label in phase1-like results.
    # We keep your current convention: is_error default 0.
    y = np.asarray([int(r.get("is_error", r.get("label", 0))) for r in rows], dtype=int)

    # Inputs
    prompts = [str(r.get("question", "")) for r in rows]
    answers = [str(r.get("model_answer", "")) for r in rows]

    # Model wrapper
    llm = LLMWrapper(a.model_name, device=a.device, max_input_tokens=a.max_input_tokens)
    try:
        if hasattr(llm, "torch_dtype"):
            llm.torch_dtype = a.dtype
    except Exception:
        pass

    tok = getattr(llm, "tokenizer", None) or transformers.AutoTokenizer.from_pretrained(a.model_name, use_fast=True)

    # Answer length (tokens)
    ans_len = np.asarray([len(tok.encode(ans, add_special_tokens=False)) for ans in answers], dtype=float)

    # LNTP/MTP (Phase-1 scorer)
    s_lntp_mean, s_mtp_mean, _stats = compute_lntp_mtp_for_qa_batch(
        llm, prompts, answers, orientation="uncertainty", return_log_stats=True
    )
    s_lntp_mean = np.asarray(s_lntp_mean, dtype=float)
    s_mtp_mean = np.asarray(s_mtp_mean, dtype=float)

    # Unnormalised (sum): mean * length
    s_lntp_sum = s_lntp_mean * ans_len
    s_mtp_sum = s_mtp_mean * ans_len

    # Correlations (score vs length)
    corr = {
        "lntp_mean_vs_len": spearman_rho(s_lntp_mean, ans_len),
        "lntp_sum_vs_len": spearman_rho(s_lntp_sum, ans_len),
        "mtp_mean_vs_len": spearman_rho(s_mtp_mean, ans_len),
        "mtp_sum_vs_len": spearman_rho(s_mtp_sum, ans_len),
    }

    # AUROC
    metrics = {}
    if np.unique(y).size >= 2:
        metrics = {
            "LNTP_mean": {"auc": compute_auroc(y, s_lntp_mean)},
            "LNTP_sum": {"auc": compute_auroc(y, s_lntp_sum)},
            "MTP_mean": {"auc": compute_auroc(y, s_mtp_mean)},
            "MTP_sum": {"auc": compute_auroc(y, s_mtp_sum)},
        }

    # -----------------------
    # Answer-span k-sweep
    # -----------------------
    k_sweep_results = None
    if a.k_list is not None and len(a.k_list) > 0:
        print("[INFO] Running answer-span k-sweep...")
        k_sweep_results = {}

        for k in a.k_list:
            truncated_answers = []
            valid = np.ones(len(answers), dtype=bool)

            for i, ans in enumerate(answers):
                ids = tok.encode(ans, add_special_tokens=False)
                ids_k = ids[:k]

                # decode WITHOUT strip first (strip can turn valid prefixes into "")
                dec = tok.decode(ids_k, skip_special_tokens=True)

                # If decode is empty/whitespace, mark invalid
                # (we do NOT want to change modeling_llm; just skip these rows)
                if dec is None or len(dec.strip()) == 0:
                    valid[i] = False
                    truncated_answers.append("")  # placeholder
                else:
                    truncated_answers.append(dec)

            idx = np.where(valid)[0]
            n_valid = int(idx.size)
            n_skipped = int((~valid).sum())

            if n_valid == 0:
                k_sweep_results[int(k)] = {
                    "n_valid": 0,
                    "n_skipped": n_skipped,
                    "LNTP_auc": None,
                    "MTP_auc": None,
                }
                continue

            # Filter
            prompts_k = [prompts[i] for i in idx]
            answers_k = [truncated_answers[i] for i in idx]
            y_k = y[idx]

            # Score
            s_lntp_k, s_mtp_k, _ = compute_lntp_mtp_for_qa_batch(
                llm,
                prompts_k,
                answers_k,
                orientation="uncertainty",
                return_log_stats=False,
            )
            s_lntp_k = np.asarray(s_lntp_k, dtype=float)
            s_mtp_k  = np.asarray(s_mtp_k, dtype=float)

            auc_lntp_k = None
            auc_mtp_k = None
            if np.unique(y_k).size >= 2:
                auc_lntp_k = compute_auroc(y_k, s_lntp_k)
                auc_mtp_k  = compute_auroc(y_k, s_mtp_k)

            k_sweep_results[int(k)] = {
                "n_valid": n_valid,
                "n_skipped": n_skipped,
                "LNTP_auc": auc_lntp_k,
                "MTP_auc": auc_mtp_k,
            }

        print("[INFO] k-sweep done. ks:", sorted(k_sweep_results.keys()))


    # Per-example output
    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(
                json.dumps(
                    {
                        "qid": r.get("qid"),
                        "label": int(y[i]),
                        "answer_len_tokens": float(ans_len[i]),
                        "lntp_mean": float(s_lntp_mean[i]),
                        "lntp_sum": float(s_lntp_sum[i]),
                        "mtp_mean": float(s_mtp_mean[i]),
                        "mtp_sum": float(s_mtp_sum[i]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "input": str(frozen_path),
        "model_name": a.model_name,
        "seed": a.seed,
        "n": int(len(rows)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "correlations_spearman": corr,
        "metrics": metrics,
        "answer_span_k_sweep": (
            {"k_list": a.k_list, "results": k_sweep_results}
            if a.k_list is not None and len(a.k_list) > 0
            else None
        ),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Always write separate k-sweep manifest if requested AND k_list provided
    if a.out_k_sweep_manifest and (a.k_list is not None and len(a.k_list) > 0):
        out_k = Path(a.out_k_sweep_manifest)
        out_k.parent.mkdir(parents=True, exist_ok=True)
        out_k.write_text(
            json.dumps(
                {
                    "input": str(frozen_path),
                    "model_name": a.model_name,
                    "seed": a.seed,
                    "k_list": a.k_list,
                    "results": k_sweep_results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"[OK] wrote {out_jsonl}")
    print(f"[OK] wrote {out_manifest}")
    if a.out_k_sweep_manifest and (a.k_list is not None and len(a.k_list) > 0):
        print(f"[OK] wrote {a.out_k_sweep_manifest}")


if __name__ == "__main__":
    main()