# src/prepare_truthfulqa_simplified.py
#
# Optional provenance script.
# Converts a TruthfulQA source file into a simplified JSON representation.
#
# This script is NOT required for reproducing Phase 1 results.
# Phase 1 uses the frozen file:
#   benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl
#
# The goal here is provenance: provide a transparent step to derive
# benchmarks/truthfulqa_simplified.json from a TruthfulQA export.
#
# Supported inputs:
#   - JSON list of records
#   - JSONL (one JSON object per line)
#
# Expected output fields per item:
#   qid, question, reference_answer
#
# Notes:
# - TruthfulQA has multiple variants; field names can differ across exports.
# - Adjust KEY_CANDIDATES if your export differs.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Field candidates (common TruthfulQA exports)
# -----------------------------

QUESTION_KEYS = ["question", "prompt", "query", "input"]
REFERENCE_KEYS = [
    "best_answer",
    "reference_answer",
    "reference",
    "answer",
    "gold_answer",
    "correct_answer",
    "best",
]

# Some exports use arrays of correct answers.
REFERENCE_LIST_KEYS = ["correct_answers", "answers_true", "best_answers"]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Heuristic: JSONL if multiple lines and first char isn't '['
    if "\n" in text and not text.lstrip().startswith("["):
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError("Unsupported JSON format. Provide a JSON list or JSONL file.")


def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def normalize_reference(d: Dict[str, Any]) -> str:
    # scalar reference
    ref = pick_first(d, REFERENCE_KEYS)
    if isinstance(ref, str) and ref.strip():
        return ref.strip()

    # list reference
    ref_list = pick_first(d, REFERENCE_LIST_KEYS)
    if isinstance(ref_list, list) and len(ref_list) > 0:
        # choose first non-empty string
        for x in ref_list:
            if isinstance(x, str) and x.strip():
                return x.strip()

    return ""


def normalize_question(d: Dict[str, Any]) -> str:
    q = pick_first(d, QUESTION_KEYS)
    if isinstance(q, str) and q.strip():
        return q.strip()
    return ""


def make_qid(i: int, d: Dict[str, Any]) -> str:
    # prefer explicit ids if present
    for k in ["qid", "id", "question_id", "idx", "index"]:
        if k in d and d[k] is not None:
            return str(d[k])
    return str(i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="TruthfulQA export (JSON or JSONL)")
    ap.add_argument(
        "--output",
        type=str,
        default="benchmarks/truthfulqa_simplified.json",
        help="Output path for simplified JSON",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    rows = read_json_or_jsonl(in_path)
    if not rows:
        raise ValueError("Input appears empty.")

    simplified: List[Dict[str, Any]] = []
    skipped = 0

    for i, r in enumerate(rows, start=1):
        if not isinstance(r, dict):
            skipped += 1
            continue

        q = normalize_question(r)
        ref = normalize_reference(r)
        if not q:
            skipped += 1
            continue

        simplified.append(
            {
                "qid": make_qid(i, r),
                "question": q,
                "reference_answer": ref,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(simplified, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Wrote: {out_path} | N={len(simplified)} | skipped={skipped}")
    print("Note: This script is optional. Phase 1 evaluation uses the frozen JSONL file.")


if __name__ == "__main__":
    main()