# src/self_iaa_prepare_round2_simple.py
#
# Prepare Self-IAA Round-2 (blind) from frozen Phase-1 dataset.
# MINIMAL version: only produces "hallucinated": null for manual labeling.

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = REPO_ROOT / "benchmarks" / "truthfulqa_hallu_frozen_model_outputs_300.jsonl"

OUT_QIDS_PATH = REPO_ROOT / "benchmarks" / "self_iaa_round2_qids_seed42_n80.json"
OUT_TEMPLATE_PATH = REPO_ROOT / "benchmarks" / "self_iaa_round2_template_seed42_n80.jsonl"

SEED = 42
N_TOTAL = 80

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[Dict[str, Any]], n_total: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)

    y = np.array([int(r["hallucinated"]) for r in rows], dtype=int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0_draw = n_total // 2
    n1_draw = n_total - n0_draw

    s0 = rng.choice(idx0, size=n0_draw, replace=False)
    s1 = rng.choice(idx1, size=n1_draw, replace=False)

    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)

    print(f"[Self-IAA Sampling] N={n_total} | hallu=0: {n0_draw} | hallu=1: {n1_draw}")
    return [rows[i] for i in idx.tolist()]

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    rows = read_jsonl(INPUT_PATH)

    y = np.array([int(r["hallucinated"]) for r in rows], dtype=int)
    print(
        f"[Phase-1 Labels] N={len(rows)} | "
        f"hallu=0: {(y==0).sum()} | hallu=1: {(y==1).sum()}"
    )

    sampled = stratified_sample(rows, N_TOTAL, SEED)

    # Freeze qids
    qids = [r["qid"] for r in sampled]
    OUT_QIDS_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_total": N_TOTAL,
                "sampling": "stratified_50_50",
                "qids": qids,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote frozen qids: {OUT_QIDS_PATH}")

    # Blind template (MINIMAL)
    rng = np.random.default_rng(SEED + 999)
    rng.shuffle(sampled)

    with OUT_TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        for r in sampled:
            out = {
                "qid": r["qid"],
                "question": r.get("question", ""),
                "reference_answer": r.get("reference_answer", ""),
                "model_answer": r["model_answer"],
                "hallucinated": None,  # <-- ONLY FIELD YOU FILL
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote blind Round-2 template: {OUT_TEMPLATE_PATH}")
    print("→ Fill 'hallucinated' with 0 or 1 (blind)")
    print("→ Save as: benchmarks/self_iaa_round2_labels_seed42_n80.jsonl")


if __name__ == "__main__":
    main()