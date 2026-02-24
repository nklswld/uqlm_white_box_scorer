"""
Convert frozen TruthfulQA model outputs into a normalized JSONL schema for downstream evaluation.
Input: a JSONL file where each line is a dict-like record (expects keys like qid, question, model_answer).
Output: a JSONL file with a fixed set of fields (qid, task, question, model_answer, gold, is_error, meta, ...).
The transformation is deterministic: it performs a pure, per-line mapping with no randomness and stable I/O paths.
Missing optional fields are handled via defaults (empty string / None) to keep the output schema consistent.
"""

# phase_2_medical/src/convert_truthfulqa_to_phase2_frozen.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # Project root for phase_2_medical (relative to this file).
REPO_ROOT = ROOT.parent                                # Repository root (one level above phase_2_medical).

IN_PATH = REPO_ROOT / "phase_1_replication" / "benchmarks" / "truthfulqa_hallu_frozen_model_outputs_300.jsonl"
OUT_PATH = ROOT / "outputs" / "frozen" / "truthfulqa_hallu_mistral_like.jsonl"



if not IN_PATH.exists():
    # Fail fast: input availability is a hard precondition for reproducible conversion runs.
    raise FileNotFoundError(f"Input not found: {IN_PATH}")

# Ensure output directory exists; writing is otherwise atomic per line but path creation is a prerequisite.
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            # Skip blank lines to avoid JSON decode errors; does not affect semantics of record-level mapping.
            continue
        obj = json.loads(line)

        # Schema normalization: map heterogeneous upstream keys into a fixed evaluation-friendly record layout.
        row = {
            "qid": str(obj.get("qid", "")),             # Normalize to string for stable joins across pipelines.
            "task": "truthfulqa",                       # Convention: explicit task tag for multi-benchmark tooling.
            "question": str(obj.get("question", "")),
            "context": "",                              # Convention: TruthfulQA uses no external context here.
            "choices": {},                              # Convention: placeholder for MC tasks; kept for schema parity.
            "model_answer": str(obj.get("model_answer", "")),
            "gold": obj.get("reference_answer", None),  # NOTE: potential issue: upstream key name may vary by dataset version.
            "pred": None,                               # Reserved for downstream model scoring/prediction fields.
            "is_error": int(obj.get("hallucinated", 0)),# Convention: 1 indicates hallucinated/error; 0 otherwise.
            "meta": {
                "source": IN_PATH.name                  # Provenance: record source file name for audit/debug.
            },
        }

        # Preserve Unicode content in questions/answers for faithful downstream analysis and review.
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[OK] Wrote: {OUT_PATH}")