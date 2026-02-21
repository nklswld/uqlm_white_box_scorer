"""Generate "frozen" Phase 2 model outputs for PubMedQA and MedQA (4-option) as JSONL.

Inputs: a prepared labeled JSONL for a single task (PubMedQA: question+context+gold; MedQA: question+choices+gold).
Outputs: JSONL with the standardized schema including raw model generations, constrained prediction, and error flag.
Designed for reproducible evaluation artifacts: deterministic by default (greedy decoding; temperature=0.0).
Reproducibility note: enabling sampling (--do_sample) intentionally breaks determinism unless the caller sets RNG seeds.
"""

# phase_2_medical/src/generate_frozen_phase2.py
# One generator for Phase 2 frozen outputs: supports PubMedQA + MedQA (4-option).
#
# Output JSONL schema (per line):
#   {
#     "qid": str,
#     "task": "pubmedqa"|"medqa",
#     "question": str,
#     "context": str,              # pubmedqa only (else "")
#     "choices": {"A":..,"B":..,"C":..,"D":..},  # medqa only (else {})
#     "gold": str,                 # pubmedqa: yes/no/maybe ; medqa: A/B/C/D
#     "model_answer": str,         # raw decoded generation (post-prompt)
#     "pred": str|None,            # extracted constrained prediction
#     "is_error": int,             # 1 if pred != gold or pred missing
#     "meta": {
#        "model": str,
#        "max_new_tokens": int,
#        "temperature": float,
#        "do_sample": bool,
#        "prompt_truncation_max_length": int,
#        "prompt_chars": int
#     }
#   }
#
# Notes:
# - Deterministic generation (default do_sample=False, temperature=0.0).
# - Strict output contracts:
#    PubMedQA -> yes/no/maybe
#    MedQA    -> A/B/C/D
# - This script uses HuggingFace Transformers (not `datasets`).

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# IO helpers
# -----------------------------
def ensure_parent_dir(p: Path) -> None:
    """Create the parent directory for a target path if needed."""
    p.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping empty lines."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Ignore blank lines to keep input tolerant to formatting.
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL (one JSON object per line)."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Task-specific prompt + parsing
# -----------------------------
VALID_PUBMEDQA = {"yes", "no", "maybe"}
VALID_MEDQA = {"A", "B", "C", "D"}

# PubMedQA: accept the first token-level match anywhere in the continuation.
# NOTE: potential issue: regex will also match "yes/no/maybe" occurring in explanations if the model violates "ONLY".
_pubmedqa_re = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
# MedQA: accept the first letter match anywhere in the continuation.
# NOTE: potential issue: regex may match option letters appearing in text (e.g., "Plan B") if generation is verbose.
_medqa_re = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def build_prompt_pubmedqa(question: str, context: str) -> str:
    """Format a PubMedQA prompt with a strict one-word output contract."""
    # Heuristic: "Final answer:" anchor reduces chatter and supports fallback stripping in decoding.
    return (
        "You are answering a medical question based on the given abstract.\n"
        "Answer using exactly one word from {yes, no, maybe}.\n"
        "Output ONLY that word.\n\n"
        f"Abstract:\n{context.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Final answer:"
    )


def build_prompt_medqa(question: str, choices: Dict[str, str]) -> str:
    """Format a MedQA prompt with a strict single-letter output contract."""
    # Conventions: choices are expected under keys 'A'..'D'; missing keys are rendered as empty strings.
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


def extract_pred_pubmedqa(text: str) -> Optional[str]:
    """Extract the constrained PubMedQA label (yes/no/maybe) from a generation."""
    t = (text or "").strip()
    m = _pubmedqa_re.search(t)  # Contract: take the first match; downstream treats None as extraction failure.
    if not m:
        return None
    pred = m.group(1).lower()
    return pred if pred in VALID_PUBMEDQA else None  # Defensive: keep schema stable even if regex changes.


def extract_pred_medqa(text: str) -> Optional[str]:
    """Extract the constrained MedQA option letter (A/B/C/D) from a generation."""
    t = (text or "").strip()
    m = _medqa_re.search(t)  # Contract: take the first match; downstream treats None as extraction failure.
    if not m:
        return None
    pred = m.group(1).upper()
    return pred if pred in VALID_MEDQA else None


# -----------------------------
# Generation
# -----------------------------
@dataclass
class GenCfg:
    """Generation configuration for producing short, constrained outputs."""
    model_name: str
    device: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    prompt_truncation_max_length: int  # tokenizer max_length for prompt truncation


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    cfg: GenCfg,
) -> List[str]:
    """Generate continuations for a batch of prompts and return prompt-stripped strings."""
    # Tokenization invariant: padding+truncation must preserve alignment prompt_i -> output_i.
    # TODO: verify: truncation does not remove the "Final answer:" anchor for long contexts/questions.
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.prompt_truncation_max_length,
    )
    enc = {k: v.to(cfg.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        use_cache=True,  # Cache improves throughput; short max_new_tokens keeps memory bounded.
    )

    # Decode model outputs to text; we then isolate the post-prompt continuation for downstream parsing.
    decoded = tok.batch_decode(out, skip_special_tokens=True)

    gens: List[str] = []
    for full_text, prompt in zip(decoded, prompts):
        # Preferred path: HF decode returns prompt+continuation verbatim, so prefix stripping is exact.
        if full_text.startswith(prompt):
            gens.append(full_text[len(prompt):].strip())
        else:
            # Fallback for tokenization/decoding mismatches: anchor on the last "Final answer:" occurrence.
            idx = full_text.lower().rfind("final answer:")
            if idx != -1:
                gens.append(full_text[idx + len("final answer:"):].strip())
            else:
                # NOTE: potential issue: if neither prefix nor anchor is found, parsing may capture reasoning text.
                gens.append(full_text.strip())
    return gens


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """CLI entrypoint: load labeled data, run batched generation, and write frozen JSONL."""
    p = argparse.ArgumentParser(description="Generate Phase-2 frozen outputs for PubMedQA or MedQA.")
    p.add_argument("--task", type=str, required=True, choices=["pubmedqa", "medqa"])
    p.add_argument("--input", type=str, required=True, help="Prepared labeled JSONL (pubmedqa or medqa).")
    p.add_argument("--output", type=str, required=True, help="Frozen output JSONL path.")
    p.add_argument("--model", type=str, required=True, help="HF model name (e.g., mistralai/Mistral-7B-Instruct-v0.2)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy).")
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--prompt_truncation_max_length", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0=all).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ensure_parent_dir(out_path)

    rows = load_jsonl(in_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]  # Debug-only: alters reported counts; does not change generation behavior.

    gen_cfg = GenCfg(
        model_name=args.model,
        device=args.device,
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        do_sample=bool(args.do_sample),
        top_p=float(args.top_p),
        prompt_truncation_max_length=int(args.prompt_truncation_max_length),
    )

    # Load model/tokenizer (Transformers); default settings aim for deterministic decoding unless sampling is enabled.
    tok = AutoTokenizer.from_pretrained(gen_cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        # Ensure padding works for batch generation; required for tokenizer(..., padding=True).
        tok.pad_token = tok.eos_token  # NOTE: potential issue: EOS-as-PAD can affect attention for some models.

    model = AutoModelForCausalLM.from_pretrained(
        gen_cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    frozen_rows: List[Dict[str, Any]] = []

    def _gold_norm(task: str, g: Any) -> str:
        """Normalize gold labels to the task's constrained alphabet; fail fast on unexpected values."""
        if task == "pubmedqa":
            gg = str(g).strip().lower()
            if gg not in VALID_PUBMEDQA:
                raise ValueError(f"Unexpected PubMedQA gold: {g!r}")  # Prevent silent metric corruption.
            return gg
        gg = str(g).strip().upper()
        if gg not in VALID_MEDQA:
            raise ValueError(f"Unexpected MedQA gold: {g!r}")  # Prevent silent metric corruption.
        return gg

    n = len(rows)
    bs = gen_cfg.batch_size

    for start in range(0, n, bs):
        batch = rows[start:start + bs]

        if args.task == "pubmedqa":
            prompts = [
                build_prompt_pubmedqa(
                    question=str(r.get("question", "")),
                    context=str(r.get("context", "")),
                )
                for r in batch
            ]
        else:
            prompts = [
                build_prompt_medqa(
                    question=str(r.get("question", "")),
                    choices=dict(r.get("choices", {})),
                )
                for r in batch
            ]

        gens = generate_batch(model=model, tok=tok, prompts=prompts, cfg=gen_cfg)

        for r, prompt, gen in zip(batch, prompts, gens):
            gold = _gold_norm(args.task, r.get("gold", ""))

            if args.task == "pubmedqa":
                pred = extract_pred_pubmedqa(gen)
                context = str(r.get("context", ""))  # Preserve original context verbatim for auditing.
                choices = {}  # Schema contract: choices empty for PubMedQA.
            else:
                pred = extract_pred_medqa(gen)
                context = ""  # Schema contract: context empty for MedQA.
                choices = dict(r.get("choices", {}))  # Preserve original choices verbatim for auditing.

            # Error definition: missing pred counts as error (extraction failure), as does mismatch vs gold.
            is_error = 1
            if pred is not None:
                is_error = 0 if pred == gold else 1

            frozen_rows.append(
                {
                    "qid": r.get("qid"),
                    "task": args.task,
                    "question": r.get("question", ""),
                    "context": context,
                    "choices": choices,
                    "gold": gold,
                    "model_answer": gen,  # Raw continuation used for downstream audits and alternative parsers.
                    "pred": pred,  # Constrained prediction (or None if extraction failed).
                    "is_error": int(is_error),
                    "meta": {
                        "model": gen_cfg.model_name,
                        "max_new_tokens": gen_cfg.max_new_tokens,
                        "temperature": gen_cfg.temperature,
                        "do_sample": gen_cfg.do_sample,
                        "top_p": gen_cfg.top_p,
                        "prompt_truncation_max_length": gen_cfg.prompt_truncation_max_length,
                        "prompt_chars": len(prompt),  # Lightweight trace for truncation/debugging.
                    },
                }
            )

        done = min(start + bs, n)
        if done % 50 == 0 or done == n:
            print(f"[INFO] {done}/{n}")

    write_jsonl(out_path, frozen_rows)
    print(f"[OK] Wrote frozen JSONL: {out_path}")


if __name__ == "__main__":
    main()