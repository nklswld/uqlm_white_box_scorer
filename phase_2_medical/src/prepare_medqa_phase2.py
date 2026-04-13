"""Prepare MedQA for Phase 2.

Draw a seeded without-replacement subsample from the chosen split, write the
benchmark JSONL, and store the sampled qids separately to freeze exact sample
membership. QIDs include split, original row index, and a question-text hash.
"""

# phase_2_medical/src/prepare_medqa_phase2.py
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def normalize_gold(label: Any) -> str:
    """Normalize the MedQA gold label to {'A','B','C','D'}; raise otherwise."""
    lab = str(label).strip().upper()
    if lab not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unexpected MedQA gold label: {label!r}")
    return lab


def normalize_choices(opts: Any) -> Dict[str, str]:
    """Normalize answer options to a dense {'A','B','C','D'} mapping of strings."""
    if not isinstance(opts, dict):
        raise ValueError(f"MedQA options must be a dict, got: {type(opts)}")
    out: Dict[str, str] = {}
    for k in ["A", "B", "C", "D"]:
        out[k] = str(opts.get(k, "")).strip()
    return out


def make_qid(split: str, original_index: int, question: str) -> str:
    """Build a stable qid from split, original row index, and question hash."""
    return f"medqa::{split}::{original_index:06d}::{sha1_12(question.strip())}"


def sample_indices(n_total: int, n: int, seed: int) -> List[int]:
    """Draw an exact-N sample without replacement from a fixed local RNG."""
    if n > n_total:
        raise ValueError(f"Requested n={n} but dataset only has n_total={n_total}.")
    rng = random.Random(seed)
    return rng.sample(range(n_total), k=n)  # without replacement


def export_medqa(
    out_jsonl: Path,
    out_qids_json: Path,
    split: str,
    n: int,
    seed: int,
) -> Tuple[int, int]:
    """Export a seeded MedQA subsample to JSONL plus a frozen qid list."""
    # NOTE: potential issue: dataset revision is not pinned; upstream updates can change sampled content for same seed.
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")[split]
    n_total = len(ds)

    picked = sample_indices(n_total=n_total, n=n, seed=seed)
    # Keep output order stable after sampling so membership and row order are both reproducible.
    picked_sorted = sorted(picked)

    ensure_parent_dir(out_jsonl)
    ensure_parent_dir(out_qids_json)

    qids: List[str] = []
    written = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx in picked_sorted:
            ex = ds[int(idx)]
            question = str(ex.get("question", "")).strip()
            choices = normalize_choices(ex.get("options", {}))

            gold_raw = ex.get("answer_idx", ex.get("answer", ""))
            gold = normalize_gold(gold_raw)

            qid = make_qid(split=split, original_index=int(idx), question=question)

            row = {
                "qid": qid,
                "question": question,
                "choices": choices,
                "gold": gold,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            qids.append(qid)
            written += 1

    out_qids_json.write_text(json.dumps(qids, indent=2, ensure_ascii=False), encoding="utf-8")
    return written, n_total


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare MedQA Phase 2 JSONL (paper-like: test split, seeded N).")
    p.add_argument("--split", choices=["train", "test"], default="test", help="Which split to export.")
    p.add_argument("--n", type=int, default=1000, help="Exact number of examples to sample (without replacement).")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducible).")

    p.add_argument(
        "--out_jsonl",
        type=str,
        default="phase_2_medical/benchmarks/medqa_test_labeled_seed42_n1000.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--out_qids_json",
        type=str,
        default="phase_2_medical/benchmarks/medqa_test_qids_seed42_n1000.json",
        help="Output JSON list of sampled qids.",
    )

    args = p.parse_args()

    out_jsonl = Path(args.out_jsonl)
    out_qids_json = Path(args.out_qids_json)

    written, n_total = export_medqa(
        out_jsonl=out_jsonl,
        out_qids_json=out_qids_json,
        split=args.split,
        n=args.n,
        seed=args.seed,
    )

    print(f"[OK] MedQA prepared: split={args.split} total={n_total} sampled={written} seed={args.seed}")
    print(f"     JSONL: {out_jsonl}")
    print(f"     QIDs:  {out_qids_json}")


if __name__ == "__main__":
    main()
