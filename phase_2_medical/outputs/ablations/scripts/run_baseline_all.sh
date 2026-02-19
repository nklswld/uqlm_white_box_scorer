#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail

# --------------------------------------------------
# Load HF token from .env (if available)
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "[INFO] Loading HF token from .env"
  set -a
  source "${REPO_ROOT}/.env"
  set +a

  # Ensure both common variable names are set
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
else
  echo "[INFO] No .env found at repo root (continuing without explicit HF token)"
fi


# -------------------------------------------------------------------
# Phase 2 Baseline Runner
# Writes baseline artifacts to: phase_2_medical/outputs/final/
#
# Produces 4 runs:
#   1) Mistral     x PubMedQA
#   2) Mistral     x MedQA
#   3) BioMistral  x PubMedQA
#   4) BioMistral  x MedQA
# -------------------------------------------------------------------

# repo root = 4 levels up from this script:
# phase_2_medical/outputs/ablations/scripts -> (up) scripts -> ablations -> outputs -> phase_2_medical -> REPO ROOT
# Anchor on phase_2_medical directory relative to this script
PHASE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "${PHASE2_DIR}/.." && pwd)"


# guard against accidental double nesting
PHASE2_DIR="$(cd "${PHASE2_DIR}" && pwd)"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"


SRC="${PHASE2_DIR}/src/run_phase2.py"
[[ -f "${SRC}" ]] || { echo "ERROR: SRC not found: ${SRC}"; exit 1; }

OUT_DIR="${PHASE2_DIR}/outputs/final"
FROZEN_DIR="${PHASE2_DIR}/outputs/frozen"
mkdir -p "${OUT_DIR}"

echo "Repo root:     ${REPO_ROOT}"
echo "Phase2 dir:    ${PHASE2_DIR}"
echo "Output dir:    ${OUT_DIR}"
echo "Frozen dir:    ${FROZEN_DIR}"
echo

# ---------------------------
# 1) Mistral × PubMedQA
# ---------------------------
echo "=== [1/4] Mistral × PubMedQA ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${SRC}" \
  --task pubmedqa \
  --frozen_jsonl "${FROZEN_DIR}/pubmedqa_mistral7b.jsonl" \
  --out_jsonl "${OUT_DIR}/pubmedqa_mistral.B5000.results.jsonl" \
  --out_manifest "${OUT_DIR}/pubmedqa_mistral.B5000.manifest.json" \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --B 5000 \
  --batch_size 4 \
  --hidden_batch_size 4 \
  --max_context_tokens 128 \
  --seed 42 \
  --n_splits 5 \
  --ci 0.95 

echo

# ---------------------------
# 2) Mistral × MedQA
# ---------------------------
echo "=== [2/4] Mistral × MedQA ==="
python "${SRC}" \
  --task medqa \
  --frozen_jsonl "${FROZEN_DIR}/medqa_mistral7b.jsonl" \
  --out_jsonl "${OUT_DIR}/medqa_mistral.B5000.results.jsonl" \
  --out_manifest "${OUT_DIR}/medqa_mistral.B5000.manifest.json" \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --B 5000 \
  --batch_size 1 \
  --hidden_batch_size 1 \
  --max_context_tokens 64 \
  --seed 42 \
  --n_splits 5 \
  --ci 0.95 

echo

# ---------------------------
# 3) BioMistral × PubMedQA
# ---------------------------
echo "=== [3/4] BioMistral × PubMedQA ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python "${SRC}" \
  --task pubmedqa \
  --frozen_jsonl "${FROZEN_DIR}/pubmedqa_biomistral7b.jsonl" \
  --out_jsonl "${OUT_DIR}/pubmedqa_biomistral.B5000.results.jsonl" \
  --out_manifest "${OUT_DIR}/pubmedqa_biomistral.B5000.manifest.json" \
  --model_name "BioMistral/BioMistral-7B" \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --B 5000 \
  --batch_size 4 \
  --hidden_batch_size 4 \
  --max_context_tokens 128 \
  --seed 42 \
  --n_splits 5 \
  --ci 0.95 

echo

# ---------------------------
# 4) BioMistral × MedQA
# ---------------------------
echo "=== [4/4] BioMistral × MedQA ==="
python "${SRC}" \
  --task medqa \
  --frozen_jsonl "${FROZEN_DIR}/medqa_biomistral7b.jsonl" \
  --out_jsonl "${OUT_DIR}/medqa_biomistral.B5000.results.jsonl" \
  --out_manifest "${OUT_DIR}/medqa_biomistral.B5000.manifest.json" \
  --model_name "BioMistral/BioMistral-7B" \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --B 5000 \
  --batch_size 1 \
  --hidden_batch_size 1 \
  --max_context_tokens 64 \
  --seed 42 \
  --n_splits 5 \
  --ci 0.95 

echo
echo "✅ Done. Baseline outputs written to: ${OUT_DIR}"