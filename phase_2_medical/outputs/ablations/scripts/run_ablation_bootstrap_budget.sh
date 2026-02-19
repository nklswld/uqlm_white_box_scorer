#!/usr/bin/env bash
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


# Ablation: Bootstrap Budget
# Vary: --B (number of bootstrap resamples)
# Goal: show CI width convergence vs B
#
# Output: outputs/ablations/bootstrap_budget/<task>_<model>/B_<B>/

# Robust path resolution (independent of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"     # -> phase_2_medical

RUN="${PHASE2_ROOT}/src/run_phase2.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/bootstrap_budget"
mkdir -p "${OUT_ROOT}"

SEED=42
CI=0.95
N_SPLITS=5

HIDDEN_LAYERS="16"
HIDDEN_POOLING="mean_answer"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tasks
TASKS=("medqa" "pubmedqa")

# Model keys
MODEL_KEYS=("mistral" "biomistral")

# Exakte HF-Modell-IDs
model_name_for_key () {
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Frozen-Dateien (outputs/frozen)
frozen_for () {
  case "$1-$2" in
    medqa-mistral)       echo "medqa_mistral7b.jsonl" ;;
    medqa-biomistral)    echo "medqa_biomistral7b.jsonl" ;;
    pubmedqa-mistral)    echo "pubmedqa_mistral7b.jsonl" ;;
    pubmedqa-biomistral) echo "pubmedqa_biomistral7b.jsonl" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Task-spezifische Defaults wie in deinen manuellen Runs
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

B_LIST=(1000 2000 5000 10000)

for TASK in "${TASKS[@]}"; do
  read -r BS HBS MAX_CTX_TOK <<<"$(task_params "${TASK}")"

  for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_NAME="$(model_name_for_key "${MODEL_KEY}")"
    FROZEN="$(frozen_for "${TASK}" "${MODEL_KEY}")"

    FROZEN_PATH="${PHASE2_ROOT}/outputs/frozen/${FROZEN}"
    if [[ ! -f "${FROZEN_PATH}" ]]; then
      echo "[ERROR] Frozen file not found: ${FROZEN_PATH}"
      exit 1
    fi

    for B in "${B_LIST[@]}"; do
      TAG="${TASK}_${MODEL_KEY}_B${B}"
      OUT_DIR="${OUT_ROOT}/${TASK}_${MODEL_KEY}/B_${B}"
      mkdir -p "${OUT_DIR}"

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