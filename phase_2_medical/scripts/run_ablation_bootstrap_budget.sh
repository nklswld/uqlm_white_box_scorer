#!/usr/bin/env bash
set -euo pipefail

# NOTE: This file is a Bash script (despite the surrounding Python-only tooling in the repo).
# It runs a bootstrap-budget ablation by repeatedly invoking `run_phase2.py` with varying `--B`.
# Inputs: repo-root `.env` (optional HF token), frozen prediction JSONL per (task, model), run config constants.
# Outputs: per-run results + manifest written under `outputs/ablations/bootstrap_budget/<task>_<model>/B_<B>/`.
# Determinism: controlled via `--seed` and fixed data split count (`--n_splits`); token loading is non-deterministic only in presence/absence of `.env`.

# --------------------------------------------------
# Optional Hugging Face auth: load token from repo-root `.env` if present.
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "[INFO] Loading HF token from .env"
  set -a
  source "${REPO_ROOT}/.env"
  set +a

  # Convention: support both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` expected by common HF tooling.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
else
  # NOTE: potential issue: private/ gated models will fail downstream without a valid HF token.
  echo "[INFO] No .env found at repo root (continuing without explicit HF token)"
fi


# Ablation: Bootstrap Budget
# Vary: --B (number of bootstrap resamples)
# Goal: quantify CI-width convergence as B increases (compute/accuracy trade-off).
#
# Output layout: outputs/ablations/bootstrap_budget/<task>_<model>/B_<B>/

# Robust path resolution (independent of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"           # -> phase_2_medical

RUN="${PHASE2_ROOT}/src/run_phase2.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/bootstrap_budget"
mkdir -p "${OUT_ROOT}"

# Global experimental controls (kept constant across all ablation runs)
SEED=42
CI=0.95
N_SPLITS=5

# Hidden-state extraction configuration (kept fixed so only bootstrap budget changes)
HIDDEN_LAYERS="16"
HIDDEN_POOLING="mean_answer"

# CUDA allocator tuning to reduce fragmentation / OOM risk for large models on long runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tasks (must match `run_phase2.py --task` accepted values)
TASKS=("medqa" "pubmedqa")

# Model keys used for concise directory naming + lookup into full HF model IDs
MODEL_KEYS=("mistral" "biomistral")

# Map stable short keys -> exact Hugging Face model identifiers.
model_name_for_key () {
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Map (task, model) -> frozen JSONL filename under `outputs/frozen/`.
# NOTE: potential issue: this hard-codes filenames; if upstream naming changes, the ablation will fail fast below.
frozen_for () {
  case "$1-$2" in
    medqa-mistral)       echo "medqa_mistral7b.jsonl" ;;
    medqa-biomistral)    echo "medqa_biomistral7b.jsonl" ;;
    pubmedqa-mistral)    echo "pubmedqa_mistral7b.jsonl" ;;
    pubmedqa-biomistral) echo "pubmedqa_biomistral7b.jsonl" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Task-specific runtime defaults aligned with prior manual runs:
# returns: batch_size hidden_batch_size max_context_tokens
task_params () {
  case "$1" in
    medqa)
      echo "1 1 64"    # BS HBS MAX_CTX_TOK
      ;;
    pubmedqa)
      echo "4 4 128"
      ;;
    *)
      echo "UNKNOWN_TASK" ; exit 1 ;;
  esac
}

# Bootstrap budgets to evaluate; interpreted by `run_phase2.py` as number of resamples.
B_LIST=(1000 2000 5000 10000)

for TASK in "${TASKS[@]}"; do
  # Invariant: (BS, HBS, MAX_CTX_TOK) are fixed per task across all models and B settings.
  read -r BS HBS MAX_CTX_TOK <<<"$(task_params "${TASK}")"

  for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_NAME="$(model_name_for_key "${MODEL_KEY}")"
    FROZEN="$(frozen_for "${TASK}" "${MODEL_KEY}")"

    # Hard dependency: frozen predictions must exist; otherwise results would be partially missing / incomparable.
    FROZEN_PATH="${PHASE2_ROOT}/outputs/frozen/${FROZEN}"
    if [[ ! -f "${FROZEN_PATH}" ]]; then
      echo "[ERROR] Frozen file not found: ${FROZEN_PATH}"
      exit 1
    fi

    for B in "${B_LIST[@]}"; do
      # Tag is used for both filenames and run identification in downstream aggregation.
      TAG="${TASK}_${MODEL_KEY}_B${B}"
      OUT_DIR="${OUT_ROOT}/${TASK}_${MODEL_KEY}/B_${B}"
      mkdir -p "${OUT_DIR}"

      # Only `--B` varies within this loop; all other parameters are held constant for fair CI-width comparison.
      python "${RUN}" \
        --task "${TASK}" \
        --frozen_jsonl "${FROZEN_PATH}" \
        --out_jsonl "${OUT_DIR}/${TAG}.results.jsonl" \
        --out_manifest "${OUT_DIR}/${TAG}.manifest.json" \
        --model_name "${MODEL_NAME}" \
        --device "cuda:0" \
        --dtype "bfloat16" \
        --seed "${SEED}" \
        --n_splits "${N_SPLITS}" \
        --B "${B}" \
        --ci "${CI}" \
        --batch_size "${BS}" \
        --hidden_batch_size "${HBS}" \
        --hidden_layers ${HIDDEN_LAYERS} \
        --hidden_pooling "${HIDDEN_POOLING}" \
        --max_context_tokens "${MAX_CTX_TOK}"
    done

    echo "[OK] Bootstrap budget ablation finished for: ${TASK} × ${MODEL_KEY}. Outputs in: ${OUT_ROOT}/${TASK}_${MODEL_KEY}/"
  done
done

echo "[OK] Bootstrap budget ablation finished. Outputs root: ${OUT_ROOT}"