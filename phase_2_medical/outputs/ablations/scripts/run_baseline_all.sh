#!/usr/bin/env bash
# Phase 2 medical baseline runner (Bash entrypoint).
# Runs four fixed configurations (model × dataset) via src/run_phase2.py.
# Inputs: optional Hugging Face token from repo-root .env; frozen JSONL inputs under outputs/frozen/.
# Outputs: per-run results JSONL + manifest JSON written under outputs/final/ (created if missing).
# Determinism: passes a fixed seed (42); reproducibility further depends on run_phase2.py and GPU kernels.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail  # Fail fast on errors, unset vars, and pipeline failures (avoids partial artifacts).

# --------------------------------------------------
# Load HF token from .env (if available)
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "[INFO] Loading HF token from .env"
  set -a  # Export all variables defined by .env into the environment for child processes.
  source "${REPO_ROOT}/.env"
  set +a

  # Map the common HF token variable name used across tools/SDKs (keep both if present).
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

# Resolve paths relative to this script to avoid dependence on the current working directory.
# Anchor on phase_2_medical directory relative to this script (scripts may be invoked from anywhere).
PHASE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="$(cd "${PHASE2_DIR}/.." && pwd)"


# Guard against accidental double nesting / symlink surprises by normalizing to absolute canonical paths.
PHASE2_DIR="$(cd "${PHASE2_DIR}" && pwd)"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"


SRC="${PHASE2_DIR}/src/run_phase2.py"
[[ -f "${SRC}" ]] || { echo "ERROR: SRC not found: ${SRC}"; exit 1; }  # Hard fail: without SRC, no run is meaningful.

OUT_DIR="${PHASE2_DIR}/outputs/final"
FROZEN_DIR="${PHASE2_DIR}/outputs/frozen"
mkdir -p "${OUT_DIR}"  # Idempotent: ensures output directory exists for all runs.

echo "Repo root:     ${REPO_ROOT}"
echo "Phase2 dir:    ${PHASE2_DIR}"
echo "Output dir:    ${OUT_DIR}"
echo "Frozen dir:    ${FROZEN_DIR}"
echo

# ---------------------------
# 1) Mistral × PubMedQA
# ---------------------------
echo "=== [1/4] Mistral × PubMedQA ==="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Re-export for safety in case caller overrides env.
python "${SRC}" \
  --task pubmedqa \
  --frozen_jsonl "${FROZEN_DIR}/pubmedqa_mistral7b.jsonl" \
  --out_jsonl "${OUT_DIR}/pubmedqa_mistral.B5000.results.jsonl" \
  --out_manifest "${OUT_DIR}/pubmedqa_mistral.B5000.manifest.json" \
  --model_name "mistralai/Mistral-7B-Instruct-v0.2" \
  --device "cuda:0" \
  --dtype "bfloat16" \
  --B 5000 \  # Bootstrap/resampling budget (B); impacts CI stability and runtime.
  --batch_size 4 \  # Runtime batch size; tuned per task to fit GPU memory.
  --hidden_batch_size 4 \  # Internal batching (if supported by SRC); keep aligned with batch_size unless justified.
  --max_context_tokens 128 \  # Task-specific context cap; too small may truncate prompts and change metrics.
  --seed 42 \  # Fixed RNG seed for reproducible resampling/splitting (subject to backend determinism).
  --n_splits 5 \  # Cross-validation / split count used by SRC (assumed); affects variance estimates.
  --ci 0.95  # Confidence level for intervals produced by SRC.

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
  --batch_size 1 \  # NOTE: potential issue: batch_size=1 may be required for memory, but can reduce throughput.
  --hidden_batch_size 1 \
  --max_context_tokens 64 \  # MedQA prompts often shorter here; truncation risk remains if upstream formatting changes.
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
echo "Done. Baseline outputs written to: ${OUT_DIR}"