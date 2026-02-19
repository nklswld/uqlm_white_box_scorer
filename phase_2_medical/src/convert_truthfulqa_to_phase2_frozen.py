import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # phase_2_medical
REPO_ROOT = ROOT.parent                                # repo root

IN_PATH = REPO_ROOT / "phase_1_replication" / "benchmarks" / "truthfulqa_hallu_frozen_model_outputs_300.jsonl"
OUT_PATH = ROOT / "outputs" / "frozen" / "truthfulqa_hallu_mistral_like.jsonl"



if not IN_PATH.exists():
    raise FileNotFoundError(f"Input not found: {IN_PATH}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        if not line.strip():
            continue
        obj = json.loads(line)

        row = {
            "qid": str(obj.get("qid", "")),
            "task": "truthfulqa",
            "question": str(obj.get("question", "")),
            "context": "",
            "choices": {},
            "model_answer": str(obj.get("model_answer", "")),
            "gold": obj.get("reference_answer", None),
            "pred": None,
            "is_error": int(obj.get("hallucinated", 0)),
            "meta": {
                "source": IN_PATH.name
            },
        }

        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[OK] Wrote: {OUT_PATH}")