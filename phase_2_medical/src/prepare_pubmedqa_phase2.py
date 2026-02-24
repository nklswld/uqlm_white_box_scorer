"""Prepare PubMedQA labeled data for Phase 2 evaluation/training.

Loads the Hugging Face dataset `qiaojin/PubMedQA` (config: `pqa_labeled`) for a given split.
Extracts a stable per-example identifier, question text, normalized abstract context, and gold label.
Writes one JSON object per line (JSONL) to the specified output path.
Determinism: output is deterministic given the dataset version and split; surrogate IDs use SHA-1 of question text.
"""

# phase_2_medical/src/prepare_pubmedqa_phase2.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
from typing import Any, Dict, Optional

from datasets import load_dataset


def sha1_12(s: str) -> str:
    """Return a stable 12-hex-character SHA-1 prefix for short surrogate identifiers."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def pick_qid(ex: Dict[str, Any]) -> str:
    """Select a stable PubMedQA example ID across dataset variants; fallback to a deterministic surrogate."""
    # PubMedQA examples can expose different ID fields depending on dataset/version/processing.
    for key in ("pubid", "pmid", "id", "question_id", "qid"):
        v = ex.get(key, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    # Fallback: deterministic surrogate keyed by question text (assumes question uniquely identifies an example).
    q = str(ex.get("question", "")).strip()
    # NOTE: potential issue: identical/empty questions across examples will collide under this surrogate scheme.
    return f"noid::{sha1_12(q)}"


def normalize_gold(label: str) -> str:
    """Normalize PubMedQA gold label to {'yes','no','maybe'}; raise on unexpected values."""
    lab = str(label).strip().lower()
    if lab not in {"yes", "no", "maybe"}:
        # Fail fast: unexpected labels would silently corrupt downstream label statistics.
        raise ValueError(f"Unexpected PubMedQA gold label: {label!r}")
    return lab


def export_pubmedqa_labeled(out_path: Path, split: str = "train") -> int:
    """
    Export PubMedQA `pqa_labeled` split to Phase-2 JSONL schema (qid, question, context, gold).
    Context is reduced to plain text by joining abstract fragments with blank lines.
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")[split]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            ex = dict(ex)

            qid = pick_qid(ex)  # Stable ID when available; deterministic surrogate otherwise.
            question = str(ex.get("question", "")).strip()
            
            context = ex.get("context", "")

            # PubMedQA "context" may be: dict with "contexts", list of fragments, or a JSON-encoded string.
            if isinstance(context, str):
                s = context.strip()
                # Heuristic parse: only attempt JSON decode when it looks like the expected {"contexts": ...} payload.
                if s.startswith("{") and '"contexts"' in s:
                    try:
                        context = json.loads(s)
                    except Exception:
                        # NOTE: potential issue: malformed/partial JSON context will pass through as raw string.
                        pass

            # Normalize context to plain text for downstream consumers expecting a single abstract string.
            if isinstance(context, dict) and "contexts" in context:
                parts = context.get("contexts", [])
                if isinstance(parts, list):
                    # Join non-empty fragments with blank lines to preserve paragraph boundaries.
                    context = "\n\n".join(str(p).strip() for p in parts if str(p).strip())
                else:
                                        # NOTE: this branch assumes non-list "contexts" payloads are rare and safely string-coercible; confirm against future dataset schema revisions.
                    context = str(parts).strip()
            elif isinstance(context, list):
                context = "\n\n".join(str(p).strip() for p in context if str(p).strip())
            else:
                context = str(context).strip()


            gold = normalize_gold(ex.get("final_decision", ""))  # PubMedQA labeled gold field.

            row = {
                "qid": qid,
                "question": question,
                "context": context,
                "gold": gold,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    return n


def main() -> None:
    """CLI entry point for exporting a PubMedQA labeled split to JSONL."""
    parser = argparse.ArgumentParser(description="Prepare PubMedQA labeled split for Phase 2 (JSONL export).")
    parser.add_argument(
        "--out",
        type=str,
        default="phase_2_medical/benchmarks/pubmedqa_labeled_phase2.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to export (PubMedQA labeled commonly provides 'train').",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    n = export_pubmedqa_labeled(out_path=out_path, split=args.split)
    print(f"[OK] Exported {n} examples to: {out_path}")


if __name__ == "__main__":
    main()