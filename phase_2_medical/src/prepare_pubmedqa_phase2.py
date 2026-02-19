# phase_2_medical/src/prepare_pubmedqa_phase2.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
from typing import Any, Dict, Optional

from datasets import load_dataset


def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def pick_qid(ex: Dict[str, Any]) -> str:
    """
    PubMedQA examples can expose different ID fields depending on the dataset version.
    We pick a stable ID if present, else fall back to a hashable surrogate.
    """
    for key in ("pubid", "pmid", "id", "question_id", "qid"):
        v = ex.get(key, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    # Fallback: use a short surrogate ID based on question text (deterministic)
    q = str(ex.get("question", "")).strip()
    return f"noid::{sha1_12(q)}"


def normalize_gold(label: str) -> str:
    lab = str(label).strip().lower()
    if lab not in {"yes", "no", "maybe"}:
        raise ValueError(f"Unexpected PubMedQA gold label: {label!r}")
    return lab


def export_pubmedqa_labeled(out_path: Path, split: str = "train") -> int:
    """
    Exports qiaojin/PubMedQA pqa_labeled to a minimal JSONL schema used by Phase 2.
    Schema per line:
      {
        "qid": ...,
        "question": ...,
        "context": ...,
        "gold": "yes|no|maybe"
      }
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")[split]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            ex = dict(ex)

            qid = pick_qid(ex)
            question = str(ex.get("question", "")).strip()
            
            context = ex.get("context", "")

            # If context is a JSON-encoded string, decode it first
            if isinstance(context, str):
                s = context.strip()
                if s.startswith("{") and '"contexts"' in s:
                    try:
                        context = json.loads(s)
                    except Exception:
                        pass

            # Now reduce to plain abstract text
            if isinstance(context, dict) and "contexts" in context:
                parts = context.get("contexts", [])
                if isinstance(parts, list):
                    context = "\n\n".join(str(p).strip() for p in parts if str(p).strip())
                else:
                    context = str(parts).strip()
            elif isinstance(context, list):
                context = "\n\n".join(str(p).strip() for p in context if str(p).strip())
            else:
                context = str(context).strip()


            gold = normalize_gold(ex.get("final_decision", ""))

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