# src/self_iaa_compute.py
#
# Self-IAA evaluation: Round 1 vs Round 2 (same annotator, blinded re-annotation)
#
# Inputs:
#   - Round 1 labels are embedded in:
#       benchmarks/truthfulqa_hallu_frozen_model_outputs_300.jsonl
#       (expects: qid, hallucinated)
#   - Round 2 labels are in:
#       benchmarks/self_iaa_round2_labels_seed42_n80.jsonl
#       (expects: qid, hallucinated)
#   - Frozen subset qids (exact evaluation set) in:
#       benchmarks/self_iaa_round2_qids_seed42_n80.json
#
# Outputs:
#   - outputs/self_iaa_summary.json

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]

R1_PATH = REPO_ROOT / "benchmarks" / "truthfulqa_hallu_frozen_model_outputs_300.jsonl"
R2_PATH = REPO_ROOT / "benchmarks" / "self_iaa_round2_labels_seed42_n80.jsonl"
QIDS_PATH = REPO_ROOT / "benchmarks" / "self_iaa_round2_qids_seed42_n80.json"

OUT_PATH = REPO_ROOT / "outputs" / "self_iaa_summary.json"

SEED = 42
BOOT_B = 5000
CI = 0.95


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels_from_jsonl(path: Path, qid_key: str = "qid", label_key: str = "hallucinated") -> Dict[str, int]:
    m: Dict[str, int] = {}
    rows = read_jsonl(path)
    for i, r in enumerate(rows, start=1):
        if qid_key not in r:
            raise KeyError(f"Missing '{qid_key}' in {path} line {i}")
        qid = str(r[qid_key])

        v = r.get(label_key, None)
        if v not in (0, 1):
            raise ValueError(f"Invalid/missing '{label_key}' for qid={qid} in {path}: {v}")
        m[qid] = int(v)

    return m


def load_freeze_qids(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "qids" in obj:
        return [str(x) for x in obj["qids"]]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError(f"Unexpected qids format in {path}")


def bootstrap_kappa_ci(y1: np.ndarray, y2: np.ndarray, seed: int, B: int, ci: float) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y1)
    kappas = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        kappas[b] = cohen_kappa_score(y1[idx], y2[idx])
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(kappas, alpha)), float(np.quantile(kappas, 1.0 - alpha))


def positive_negative_agreement(cm_2x2: np.ndarray) -> Tuple[float, float]:
    """
    Positive/Negative Agreement for binary labels (R1 rows, R2 cols):
        cm = [[a, b],
              [c, d]]
    where:
      a = both 0
      d = both 1
      b = R1=0, R2=1
      c = R1=1, R2=0

    Negative agreement (for class 0):
      P_neg = 2a / (2a + b + c)

    Positive agreement (for class 1):
      P_pos = 2d / (2d + b + c)
    """
    a, b = cm_2x2[0, 0], cm_2x2[0, 1]
    c, d = cm_2x2[1, 0], cm_2x2[1, 1]
    denom = (b + c)
    p_neg = (2 * a) / (2 * a + denom) if (2 * a + denom) > 0 else float("nan")
    p_pos = (2 * d) / (2 * d + denom) if (2 * d + denom) > 0 else float("nan")
    return float(p_pos), float(p_neg)


def main() -> None:
    if not R1_PATH.exists():
        raise FileNotFoundError(f"Missing R1 file: {R1_PATH}")
    if not R2_PATH.exists():
        raise FileNotFoundError(f"Missing R2 file: {R2_PATH}")
    if not QIDS_PATH.exists():
        raise FileNotFoundError(f"Missing QIDs file: {QIDS_PATH}")

    # Round-1 labels are embedded in the frozen outputs file
    r1 = load_labels_from_jsonl(R1_PATH, qid_key="qid", label_key="hallucinated")
    r2 = load_labels_from_jsonl(R2_PATH, qid_key="qid", label_key="hallucinated")
    qids = load_freeze_qids(QIDS_PATH)

    # Alignment check
    missing_r1 = [q for q in qids if q not in r1]
    missing_r2 = [q for q in qids if q not in r2]
    if missing_r1:
        raise ValueError(f"R1 missing labels for {len(missing_r1)} qids (first 10): {missing_r1[:10]}")
    if missing_r2:
        raise ValueError(f"R2 missing labels for {len(missing_r2)} qids (first 10): {missing_r2[:10]}")

    y1 = np.array([r1[q] for q in qids], dtype=int)
    y2 = np.array([r2[q] for q in qids], dtype=int)

    n = int(len(qids))
    agreement = float((y1 == y2).mean())
    kappa = float(cohen_kappa_score(y1, y2))

    cm = confusion_matrix(y1, y2, labels=[0, 1])  # rows=R1, cols=R2
    p_pos, p_neg = positive_negative_agreement(cm)

    low, high = bootstrap_kappa_ci(y1, y2, seed=SEED, B=BOOT_B, ci=CI)

    print("=== Self-IAA (Round 1 vs Round 2) ===")
    print(f"N = {n}")
    print(f"Agreement = {agreement:.4f}")
    print(f"Cohen's kappa = {kappa:.4f}")
    print("Confusion matrix (rows=R1, cols=R2, labels=[0,1]):")
    print(cm)
    print(f"Positive agreement (class=1) = {p_pos:.4f}")
    print(f"Negative agreement (class=0) = {p_neg:.4f}")
    print(f"Kappa {int(CI*100)}% CI (bootstrap, B={BOOT_B}) = [{low:.4f}, {high:.4f}]")

    out = {
        "n": n,
        "agreement": agreement,
        "cohen_kappa": kappa,
        "kappa_ci": {"level": CI, "low": low, "high": high, "B": BOOT_B, "seed": SEED},
        "confusion_matrix_labels_0_1": cm.tolist(),
        "positive_agreement_class_1": p_pos,
        "negative_agreement_class_0": p_neg,
        "paths": {
            "round1_labels_embedded_in": str(R1_PATH),
            "round2_labels": str(R2_PATH),
            "qids_freeze": str(QIDS_PATH),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote summary: {OUT_PATH}")


if __name__ == "__main__":
    main()