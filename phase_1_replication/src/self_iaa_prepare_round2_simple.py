"""Prepare Self-IAA Round-2 (blind) labeling artifacts from a frozen Phase-1 dataset.
Reads a JSONL of model outputs with Phase-1 labels and draws a deterministic, stratified sample.
Inputs: Phase-1 JSONL at INPUT_PATH with at least {"qid", "model_answer", "hallucinated"}.
Outputs: (1) frozen qid list JSON (OUT_QIDS_PATH) and (2) blind labeling template JSONL (OUT_TEMPLATE_PATH).
Sampling is 50/50 by Phase-1 "hallucinated" label; template order is independently shuffled.
Deterministic/reproducible given SEED, N_TOTAL, and an unchanged input file content/order.
"""

# src/self_iaa_prepare_round2_simple.py
#
# NOTE: potential issue: reproducibility depends on stable input ordering within INPUT_PATH (sampling indexes rows).

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
    """Read non-empty JSONL lines into a list of dict rows."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # ignore blank/whitespace-only lines to avoid JSON decode errors
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[Dict[str, Any]], n_total: int, seed: int) -> List[Dict[str, Any]]:
    """Deterministic 50/50 stratified sample by integer-cast r["hallucinated"] in {0,1}."""
    rng = np.random.default_rng(seed)

    # Convention: Phase-1 "hallucinated" is treated as a binary label; any non-{0,1} value will raise via int().
    y = np.array([int(r["hallucinated"]) for r in rows], dtype=int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    # Split rule: floor(n/2) negatives, remainder positives (keeps exact n_total even when odd).
    n0_draw = n_total // 2
    n1_draw = n_total - n0_draw

    # NOTE: potential issue: will error if a class has fewer than the requested draw without replacement.
    s0 = rng.choice(idx0, size=n0_draw, replace=False)
    s1 = rng.choice(idx1, size=n1_draw, replace=False)

    idx = np.concatenate([s0, s1])
    rng.shuffle(idx)  # randomize within-sample order while preserving deterministic membership

    print(f"[Self-IAA Sampling] N={n_total} | hallu=0: {n0_draw} | hallu=1: {n1_draw}")
    return [rows[i] for i in idx.tolist()]

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run Phase-1 summary, stratified sampling, and emit frozen qids + blind labeling template."""
    rows = read_jsonl(INPUT_PATH)

    # Quick sanity summary: counts are computed from the same int-cast used for sampling.
    y = np.array([int(r["hallucinated"]) for r in rows], dtype=int)
    print(
        f"[Phase-1 Labels] N={len(rows)} | "
        f"hallu=0: {(y==0).sum()} | hallu=1: {(y==1).sum()}"
    )

    sampled = stratified_sample(rows, N_TOTAL, SEED)

    # Freeze qids: canonical record of membership for Round-2, independent of any later shuffling.
    qids = [r["qid"] for r in sampled]
    OUT_QIDS_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_total": N_TOTAL,
                "sampling": "stratified_50_50",  # convention label for downstream bookkeeping
                "qids": qids,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote frozen qids: {OUT_QIDS_PATH}")

    # Blind template: use a distinct RNG stream (SEED+999) so membership and presentation order are decoupled.
    rng = np.random.default_rng(SEED + 999)
    rng.shuffle(sampled)

    with OUT_TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        for r in sampled:
            out = {
                "qid": r["qid"],
                "question": r.get("question", ""),  # tolerate missing keys; blanks preserve JSONL schema
                "reference_answer": r.get("reference_answer", ""),
                "model_answer": r["model_answer"],  # required field for labeling; missing key should fail loudly
                "hallucinated": None,  # <-- ONLY FIELD YOU FILL (blind labels: 0/1)
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote blind Round-2 template: {OUT_TEMPLATE_PATH}")
    print("→ Fill 'hallucinated' with 0 or 1 (blind)")
    print("→ Save as: benchmarks/self_iaa_round2_labels_seed42_n80.jsonl")


if __name__ == "__main__":
    main()