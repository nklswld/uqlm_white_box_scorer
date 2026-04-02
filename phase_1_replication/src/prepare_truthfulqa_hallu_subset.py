"""
Create the deterministic TruthfulQA Phase-1 subset membership artifact.

Key input:
- benchmarks/truthfulqa_simplified.json

Key output:
- benchmarks/truthfulqa_hallu_subset_300_qids.json

Purpose:
- Draw the reproducible 300-example TruthfulQA subset used in Phase 1
- Persist the exact sampled QIDs for auditability and reproducibility

Design note:
This script no longer creates a separate label template JSONL file, because the
manual hallucination labeling is now performed directly in
`benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SEED = 42
N_SUBSET = 300

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = REPO_ROOT / "benchmarks"

DATA_PATH = BENCHMARK_DIR / "truthfulqa_simplified.json"
QIDS_OUT_PATH = BENCHMARK_DIR / "truthfulqa_hallu_subset_300_qids.json"


def extract_qid(item: dict[str, Any]) -> str:
    """
    Extract a stable question identifier from a TruthfulQA item.

    Supported schemas:
    - older: {"id", "question", "best_answer"}
    - newer: {"qid", "question", "reference_answer"}

    Fallback:
    - question text, if no explicit identifier is present
    """
    for key in ("qid", "id"):
        if key in item and item[key] is not None and str(item[key]).strip():
            return str(item[key])
    return str(item.get("question", ""))


def load_truthfulqa(path: Path) -> list[dict[str, Any]]:
    """
    Load the simplified TruthfulQA benchmark export.

    The script accepts both historical field variants used in the project:
    - question + best_answer
    - question + reference_answer
    """
    if not path.exists():
        raise FileNotFoundError(f"TruthfulQA file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected truthfulqa_simplified.json to be a JSON list.")

    if not data:
        raise ValueError("truthfulqa_simplified.json is empty.")

    for idx, item in enumerate(data[:10]):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not a JSON object.")
        if "question" not in item:
            raise ValueError(
                f"Invalid schema at index {idx}: missing required key 'question'. "
                f"Got keys: {list(item.keys())}"
            )
        if "best_answer" not in item and "reference_answer" not in item:
            raise ValueError(
                f"Invalid schema at index {idx}: expected either 'best_answer' or "
                f"'reference_answer'. Got keys: {list(item.keys())}"
            )

    return data


def validate_unique_qids(qids: list[str]) -> None:
    """
    Ensure that all QIDs are unique before sampling.

    Exact reproducibility requires a one-to-one mapping between examples and identifiers.
    """
    if len(set(qids)) == len(qids):
        return

    seen: set[str] = set()
    duplicates: list[str] = []
    for qid in qids:
        if qid in seen:
            duplicates.append(qid)
        else:
            seen.add(qid)

    raise ValueError(
        "QIDs are not unique in the dataset, so reproducible subset sampling is not possible. "
        f"Example duplicates: {duplicates[:10]}"
    )


def sample_qids(items: list[dict[str, Any]], n_subset: int, seed: int) -> list[str]:
    """
    Draw an exact-N random subset without replacement using a fixed seed.
    """
    qids = [extract_qid(item) for item in items]
    validate_unique_qids(qids)

    if n_subset > len(qids):
        raise ValueError(
            f"Requested subset size {n_subset}, but dataset contains only {len(qids)} items."
        )

    rng = random.Random(seed)
    sampled_qids = qids.copy()
    rng.shuffle(sampled_qids)
    return sampled_qids[:n_subset]


def write_qids_json(path: Path, qids: list[str], seed: int, n_subset: int) -> None:
    """
    Persist sampled QIDs together with minimal provenance metadata.
    """
    payload = {
        "dataset": "truthfulqa_simplified.json",
        "sampling": "random_without_replacement",
        "n": n_subset,
        "seed": seed,
        "qids": qids,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Create the reproducible 300-example TruthfulQA subset used in Phase 1.
    """
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    items = load_truthfulqa(DATA_PATH)
    qids_subset = sample_qids(items, n_subset=N_SUBSET, seed=SEED)
    write_qids_json(QIDS_OUT_PATH, qids_subset, seed=SEED, n_subset=N_SUBSET)

    print("✅ Created TruthfulQA Phase-1 subset membership artifact.")
    print(f"Output: {QIDS_OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Subset size: {len(qids_subset)}")
    print(f"Seed: {SEED}")


if __name__ == "__main__":
    main()