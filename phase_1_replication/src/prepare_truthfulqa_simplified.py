"""Prepare a simplified TruthfulQA JSON for downstream evaluation/provenance.

Reads a TruthfulQA export file (JSON list, JSONL, or {"data": [...]} wrapper) and
normalizes it into a compact list of {qid, question, reference_answer} records.
Input: --input path to an export; output: --output path to the simplified JSON.
Determinism: conversion is deterministic given the input file and key-candidate lists;
no randomness, timestamps, or environment-dependent ordering is introduced.
"""

# phase_1_replication/src/prepare_truthfulqa_simplified.py
# Optional provenance helper: derive a simplified TruthfulQA JSON from an export.
# Not required for Phase 1 reproduction (Phase 1 uses the frozen JSONL file).
# Adjust *_KEYS below if your TruthfulQA export uses different field names.

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Field candidates (common TruthfulQA exports)
# -----------------------------
# NOTE: potential issue: TruthfulQA exports vary by source/version; update candidates if fields differ.

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

# Some exports store multiple correct answers as arrays; we select the first non-empty string.
REFERENCE_LIST_KEYS = ["correct_answers", "answers_true", "best_answers"]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a TruthfulQA export from JSON list, JSONL, or {"data": [...]} wrapper."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Heuristic: treat as JSONL when there are multiple lines and the payload is not a JSON list.
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
    """Return the first present, non-empty candidate value from a dict."""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def normalize_reference(d: Dict[str, Any]) -> str:
    """Extract a reference answer string from common scalar/list fields (best-effort)."""
    # Priority: scalar reference fields; preserves exact text apart from surrounding whitespace.
    ref = pick_first(d, REFERENCE_KEYS)
    if isinstance(ref, str) and ref.strip():
        return ref.strip()

    # Fallback: list-of-answers fields; selects first non-empty string for stability.
    ref_list = pick_first(d, REFERENCE_LIST_KEYS)
    if isinstance(ref_list, list) and len(ref_list) > 0:
        # choose first non-empty string
        for x in ref_list:
            if isinstance(x, str) and x.strip():
                return x.strip()

    # Empty string is a deliberate sentinel: downstream can distinguish "missing ref" from missing question.
    return ""


def normalize_question(d: Dict[str, Any]) -> str:
    """Extract the question/prompt string from common export fields."""
    q = pick_first(d, QUESTION_KEYS)
    if isinstance(q, str) and q.strip():
        return q.strip()
    return ""


def make_qid(i: int, d: Dict[str, Any]) -> str:
    """Choose a stable question identifier, preferring explicit ids over row index."""
    # Convention: prefer explicit IDs to preserve upstream identity across re-exports/shuffles.
    for k in ["qid", "id", "question_id", "idx", "index"]:
        if k in d and d[k] is not None:
            return str(d[k])
    return str(i)


def main() -> None:
    """CLI entry point: read export, normalize fields, write simplified JSON."""
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
            # Silent data-quality filter: items without a usable question are excluded from the benchmark.
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