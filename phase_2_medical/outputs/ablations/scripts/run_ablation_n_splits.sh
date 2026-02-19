#!/usr/bin/env bash
set -euo pipefail

# Ablation: OOF Robustness – n_splits
# Vary: --n_splits
# Fixed: seed, model, frozen inputs, B, hidden settings, etc.
#
# Output: outputs/ablations/n_splits/<task>_<model_key>/n_<n_splits>/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"     # -> phase_2_medical

RUN="${PHASE2_ROOT}/src/run_phase2.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/n_splits"
mkdir -p "${OUT_ROOT}"

SEED=42
B=5000
CI=0.95

# Keep hidden settings fixed (match your default ablations)
HIDDEN_LAYERS="16"
HIDDEN_POOLING="mean_answer"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASKS=("medqa" "pubmedqa")
MODEL_KEYS=("mistral" "biomistral")

# The actual ablation
N_SPLITS_LIST=(3 5 10)

model_name_for_key () {
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

frozen_for () {
  case "$1-$2" in
    medqa-mistral)       echo "medqa_mistral7b.jsonl" ;;
    medqa-biomistral)    echo "medqa_biomistral7b.jsonl" ;;
    pubmedqa-mistral)    echo "pubmedqa_mistral7b.jsonl" ;;
    pubmedqa-biomistral) echo "pubmedqa_biomistral7b.jsonl" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

task_params () {
  case "$1" in
    medqa)    echo "1 1 64" ;;   # BS HBS MAX_CTX_TOK
    pubmedqa) echo "4 4 128" ;;
    *) echo "UNKNOWN_TASK" ; exit 1 ;;
  esac
}

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

    for NS in "${N_SPLITS_LIST[@]}"; do
      TAG="${TASK}_${MODEL_KEY}_nsplits${NS}"
      OUT_DIR="${OUT_ROOT}/${TASK}_${MODEL_KEY}/n_${NS}"
      mkdir -p "${OUT_DIR}"

      python "${RUN}" \
        --task "${TASK}" \
        --frozen_jsonl "${FROZEN_PATH}" \
        --out_jsonl "${OUT_DIR}/${TAG}.B${B}.results.jsonl" \
        --out_manifest "${OUT_DIR}/${TAG}.B${B}.manifest.json" \
        --model_name "${MODEL_NAME}" \
        --device "cuda:0" \
        --dtype "bfloat16" \
        --seed "${SEED}" \
        --n_splits "${NS}" \
        --B "${B}" \
        --ci "${CI}" \
        --batch_size "${BS}" \
        --hidden_batch_size "${HBS}" \
        --hidden_layers ${HIDDEN_LAYERS} \
        --hidden_pooling "${HIDDEN_POOLING}" \
        --max_context_tokens "${MAX_CTX_TOK}"
    done

    echo "[OK] Finished: ${TASK} × ${MODEL_KEY} (n_splits: ${N_SPLITS_LIST[*]}). Outputs in: ${OUT_ROOT}/${TASK}_${MODEL_KEY}/"
  done
done

echo "[OK] n_splits ablation finished. Outputs root: ${OUT_ROOT}"