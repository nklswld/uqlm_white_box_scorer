"""
Build the Phase-1 TruthfulQA hallucination annotation base artifact.

Key inputs:
- benchmarks/truthfulqa_simplified.json
- benchmarks/truthfulqa_hallu_subset_300_qids.json

Key output:
- benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl

Purpose:
- Load the reproducible 300-example TruthfulQA subset
- Generate deterministic frozen model answers
- Write the canonical JSONL file that is subsequently annotated manually

Design note:
This script intentionally writes `hallucinated = null` for every row. The file is meant
to serve as the annotation base that will later be labeled manually and then used as the
final frozen Phase-1 benchmark artifact.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure phase_1_replication/ is on PYTHONPATH when running as a script
THIS_FILE = Path(__file__).resolve()
PHASE1_ROOT = THIS_FILE.parents[1]
if str(PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE1_ROOT))

from src.modeling_llm import LLMWrapper


SEED = 42
N_SUBSET = 300
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = None

REPO_ROOT = PHASE1_ROOT
BENCHMARK_DIR = REPO_ROOT / "benchmarks"

DATA_PATH = BENCHMARK_DIR / "truthfulqa_simplified.json"
SUBSET_PATH = BENCHMARK_DIR / "truthfulqa_hallu_subset_300_qids.json"
OUT_PATH = BENCHMARK_DIR / "truthfulqa_hallu_frozen_model_outputs_300.jsonl"


def set_global_seeds(seed: int = 42) -> None:
    """Set all relevant random seeds for deterministic generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subset_qids(path: Path, expected_n: int = 300) -> list[str]:
    """Load the frozen subset membership file and validate its basic structure."""
    if not path.exists():
        raise FileNotFoundError(f"Missing subset file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if not isinstance(obj, dict) or "qids" not in obj:
        raise ValueError("Subset file must be a JSON object with key 'qids'.")

    qids = obj["qids"]
    if not isinstance(qids, list) or not all(isinstance(q, str) for q in qids):
        raise ValueError("'qids' must be a list[str].")

    if len(qids) != expected_n:
        raise ValueError(f"Expected {expected_n} qids, got {len(qids)}.")

    if len(set(qids)) != len(qids):
        raise ValueError("Subset qids contain duplicates.")

    return qids


def load_truthfulqa_items(path: Path) -> list[dict[str, Any]]:
    """Load the simplified TruthfulQA benchmark export."""
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("truthfulqa_simplified.json must be a JSON list.")

    if not data:
        raise ValueError("truthfulqa_simplified.json is empty.")

    return data


def extract_qid(item: dict[str, Any]) -> str:
    """
    Extract a stable qid from a TruthfulQA item.

    Supports both historical schemas:
    - older: {'id', 'question', 'best_answer'}
    - newer: {'qid', 'question', 'reference_answer'}
    """
    for key in ("qid", "id"):
        if key in item and item[key] is not None and str(item[key]).strip():
            return str(item[key])
    return str(item.get("question", ""))


def extract_reference_answer(item: dict[str, Any]) -> str:
    """Support both historical reference-answer field names."""
    for key in ("reference_answer", "best_answer"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


@torch.no_grad()
def generate_answer(llm: LLMWrapper, question: str, max_new_tokens: int = 64) -> str:
    """
    Generate a deterministic answer using the wrapped language model.
    """
    prompt = llm.build_prompt(question)

    inputs = llm.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=(llm.max_input_tokens is not None),
        max_length=llm.max_input_tokens,
    )
    inputs = {key: value.to(llm.input_device) for key, value in inputs.items()}

    output_ids = llm.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=llm.tokenizer.pad_token_id,
    )

    full_text = llm.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    marker = "Answer:"
    if marker in full_text:
        return full_text.split(marker, maxsplit=1)[-1].strip()
    return full_text.strip()


def main() -> None:
    set_global_seeds(SEED)

    subset_qids = load_subset_qids(SUBSET_PATH, expected_n=N_SUBSET)
    items = load_truthfulqa_items(DATA_PATH)

    # Build qid index over simplified TruthfulQA
    by_qid: dict[str, dict[str, Any]] = {}
    for item in items:
        if "question" not in item:
            continue
        qid = extract_qid(item)
        if not qid.strip():
            continue
        by_qid[qid] = item

    missing = [qid for qid in subset_qids if qid not in by_qid]
    if missing:
        raise ValueError(
            f"Subset contains qids not found in truthfulqa_simplified.json. "
            f"First 10: {missing[:10]}"
        )

    llm = LLMWrapper(
        model_name=MODEL_NAME,
        max_input_tokens=MAX_INPUT_TOKENS,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for qid in subset_qids:
            item = by_qid[qid]
            question = item["question"]
            reference_answer = extract_reference_answer(item)
            model_answer = generate_answer(llm, question, max_new_tokens=MAX_NEW_TOKENS)

            row = {
                "qid": qid,
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer,
                "hallucinated": None,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ Built TruthfulQA hallucination annotation base.")
    print(f"Output: {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Items: {len(subset_qids)}")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_NAME}")
    print("Note: `hallucinated` is initialized as null and must be labeled manually.")


if __name__ == "__main__":
    main()