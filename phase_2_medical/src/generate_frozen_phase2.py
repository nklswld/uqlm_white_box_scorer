"""Generate frozen Phase-2 model outputs for PubMedQA and MedQA (4-option) as JSONL.
Inputs: task-specific labeled JSONL (PubMedQA: question, context, gold; MedQA: question, choices, gold).
Outputs: standardized JSONL with raw generation, constrained prediction (or None), and an error indicator.
Primary use: reproducible evaluation artifacts; prompt template + parsing are fixed for auditability.
Determinism: greedy decoding by default (do_sample=False, temperature=0.0); sampling intentionally breaks it.
Reproducibility note: if sampling is enabled, callers must set RNG seeds externally for repeatable runs.
"""

# phase_2_medical/src/generate_frozen_phase2.py
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
    """Ensure the parent directory of a file path exists (mkdir -p)."""
    p.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL as a list of dicts; blank lines are ignored."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Tolerate formatting-only blanks without shifting record indices.
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write dict rows as JSONL (one JSON object per line; UTF-8, no ASCII escaping)."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Task-specific prompt + parsing
# -----------------------------
VALID_PUBMEDQA = {"yes", "no", "maybe"}
VALID_MEDQA = {"A", "B", "C", "D"}

# Parsing convention: extract the *first* valid token/letter match in the decoded continuation.
# NOTE: potential issue: matches inside explanations will be accepted if the model violates "Output ONLY ...".
_pubmedqa_re = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
# NOTE: potential issue: verbose generations may yield incidental matches (e.g., "Plan B") before the intended answer.
_medqa_re = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def build_prompt_pubmedqa(question: str, context: str) -> str:
    """Build a PubMedQA prompt enforcing a one-word label output."""
    # Heuristic: keep a stable "Final answer:" anchor to support robust prompt-stripping across decoders.
    return (
        "You are answering a medical question based on the given abstract.\n"
        "Answer using exactly one word from {yes, no, maybe}.\n"
        "Output ONLY that word.\n\n"
        f"Abstract:\n{context.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Final answer:"
    )


def build_prompt_medqa(question: str, choices: Dict[str, str]) -> str:
    """Build a MedQA prompt enforcing a single-letter option output."""
    # Convention: choices are keyed by 'A'..'D'; missing keys are treated as empty strings (auditable via output JSONL).
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
    """Return the first PubMedQA label found in the generation, else None."""
    t = (text or "").strip()
    m = _pubmedqa_re.search(t)  # Invariant: first-match extraction; downstream treats None as extraction failure.
    if not m:
        return None
    pred = m.group(1).lower()
    return pred if pred in VALID_PUBMEDQA else None  # Defensive: keep schema stable if regex is edited later.


def extract_pred_medqa(text: str) -> Optional[str]:
    """Return the first MedQA option letter found in the generation, else None."""
    t = (text or "").strip()
    m = _medqa_re.search(t)  # Invariant: first-match extraction; downstream treats None as extraction failure.
    if not m:
        return None
    pred = m.group(1).upper()
    return pred if pred in VALID_MEDQA else None


# -----------------------------
# Generation
# -----------------------------
@dataclass
class GenCfg:
    """Minimal generation config for short constrained continuations."""
    model_name: str
    device: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    prompt_truncation_max_length: int  # Token-level prompt truncation cap (affects contract anchor preservation).


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    cfg: GenCfg,
) -> List[str]:
    """Generate prompt continuations and return prompt-stripped strings (one per prompt)."""
    # Alignment invariant: tokenization must preserve prompt_i -> output_i correspondence under padding/truncation.
    # NOTE: ensure prompt truncation preserves the "Final answer:" anchor; otherwise prefix stripping may fail for long inputs.
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
        use_cache=True,  # Throughput optimization; short continuations limit KV-cache growth.
    )

    # Decode full sequences; downstream parsing operates on the post-prompt continuation only.
    decoded = tok.batch_decode(out, skip_special_tokens=True)

    gens: List[str] = []
    for full_text, prompt in zip(decoded, prompts):
        # Preferred contract: decoded text is prompt + continuation, enabling exact prefix stripping.
        if full_text.startswith(prompt):
            gens.append(full_text[len(prompt):].strip())
        else:
            # Fallback: locate the last anchor occurrence to isolate the continuation despite decode mismatches.
            idx = full_text.lower().rfind("final answer:")
            if idx != -1:
                gens.append(full_text[idx + len("final answer:"):].strip())
            else:
                # NOTE: potential issue: without prefix/anchor, continuation may include reasoning and spuriously match regex.
                gens.append(full_text.strip())
    return gens


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Run deterministic (by default) batched generation and write standardized frozen JSONL."""
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
        rows = rows[: args.limit]  # Debug-only: changes reported N and output coverage (artifact is not full-dataset).

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

    # Determinism note: model.generate is deterministic under greedy decoding, but depends on external RNG when sampling.
    tok = AutoTokenizer.from_pretrained(gen_cfg.model_name, use_fast=True)
    if tok.pad_token_id is None:
        # Required for padding=True; choose a stable token to avoid tokenizer errors in batched generation.
        tok.pad_token = tok.eos_token  # NOTE: potential issue: EOS-as-PAD can change attention behavior for some models.

    model = AutoModelForCausalLM.from_pretrained(
        gen_cfg.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    frozen_rows: List[Dict[str, Any]] = []

    def _gold_norm(task: str, g: Any) -> str:
        """Normalize gold to the task alphabet; raise on unexpected labels to avoid silent metric corruption."""
        if task == "pubmedqa":
            gg = str(g).strip().lower()
            if gg not in VALID_PUBMEDQA:
                raise ValueError(f"Unexpected PubMedQA gold: {g!r}")  # Fail fast: schema violations must be explicit.
            return gg
        gg = str(g).strip().upper()
        if gg not in VALID_MEDQA:
            raise ValueError(f"Unexpected MedQA gold: {g!r}")  # Fail fast: schema violations must be explicit.
        return gg

    n = len(rows)
    bs = gen_cfg.batch_size

    for start in range(0, n, bs):
        batch = rows[start:start + bs]

        # Task switch controls both prompt construction and the downstream constrained parser.
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

            # Schema invariant: always emit both fields; the task-irrelevant one is empty for uniform downstream parsing.
            if args.task == "pubmedqa":
                pred = extract_pred_pubmedqa(gen)
                context = str(r.get("context", ""))  # Preserve verbatim to support audits of truncation/prompt quality.
                choices = {}  # Schema contract: choices empty for PubMedQA.
            else:
                pred = extract_pred_medqa(gen)
                context = ""  # Schema contract: context empty for MedQA.
                choices = dict(r.get("choices", {}))  # Preserve verbatim to support audits of option rendering.

            # Error policy: extraction failure (pred=None) is treated as an error, not as an abstention.
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
                    "model_answer": gen,  # Raw continuation for audits and alternative parsers (do not post-process here).
                    "pred": pred,  # Constrained prediction (None indicates parsing/contract violation).
                    "is_error": int(is_error),
                    "meta": {
                        "model": gen_cfg.model_name,
                        "max_new_tokens": gen_cfg.max_new_tokens,
                        "temperature": gen_cfg.temperature,
                        "do_sample": gen_cfg.do_sample,
                        "top_p": gen_cfg.top_p,
                        "prompt_truncation_max_length": gen_cfg.prompt_truncation_max_length,
                        "prompt_chars": len(prompt),  # Proxy signal for truncation/debugging without storing full prompt.
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