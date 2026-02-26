#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Hidden-probe ablation: sweep representation depth (--hidden_layers) for each (task, model).
# Inputs: repo-root .env (optional HF token), frozen predictions JSONL, task/model/layer sweep config.
# Outputs: per-layer results JSONL + manifest JSON, written under outputs/ablations/hidden_layers/.
# Determinism: fixed seed and split count are passed through to the Python runner; bootstrap B is fixed.
# NOTE: potential issue: this script is Bash despite the "Python code" label; do not run via Python.
# --------------------------------------------------

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

  # Canonicalize token name expected by Hugging Face tooling (accept HF_TOKEN as alias).
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
else
  # NOTE: continuing without a token may trigger rate limits / gated-model failures at runtime.
  echo "[INFO] No .env found at repo root (continuing without explicit HF token)"
fi


# Ablation: Hidden Probe – Layer Sweep
# Vary: --hidden_layers
# Fixed: pooling, normalize, model, frozen, B, seed, n_splits
#
# Output: outputs/ablations/hidden_layers/<task>_<model>/layer_<L>/

# Robust path resolution (independent of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"     # -> phase_2_medical

RUN="${PHASE2_ROOT}/src/run_phase2.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/hidden_layers"
mkdir -p "${OUT_ROOT}"

SEED=42
B=5000
CI=0.95
N_SPLITS=5

# Hidden-state pooling strategy used for the probe representation (kept fixed across the sweep).
HIDDEN_POOLING="mean_answer"

# CUDA allocator hint for long-running inference; reduces fragmentation in some workloads.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tasks
TASKS=("medqa" "pubmedqa")

# Model keys
MODEL_KEYS=("mistral" "biomistral")

# Map short model keys (used in filenames/paths) to exact Hugging Face model IDs.
model_name_for_key () {
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Frozen evaluation inputs (precomputed model outputs) expected under outputs/frozen/.
# Invariant: must exist before the sweep; missing files are treated as hard errors.
frozen_for () {
  case "$1-$2" in
    medqa-mistral)       echo "medqa_mistral7b.jsonl" ;;
    medqa-biomistral)    echo "medqa_biomistral7b.jsonl" ;;
    pubmedqa-mistral)    echo "pubmedqa_mistral7b.jsonl" ;;
    pubmedqa-biomistral) echo "pubmedqa_biomistral7b.jsonl" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Task-specific runtime parameters mirroring the manual baseline runs.
# Convention: echo "<batch_size> <hidden_batch_size> <max_context_tokens>".
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

# Layer indices to probe. Assumed valid for all selected model backbones.
# TODO: verify: requested layers exist for each model; invalid indices may fail inside run_phase2.py.
LAYERS=(4 16 24 32)

for TASK in "${TASKS[@]}"; do
  # Parse task defaults into positional variables consumed by the runner CLI.
  read -r BS HBS MAX_CTX_TOK <<<"$(task_params "${TASK}")"

  for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_NAME="$(model_name_for_key "${MODEL_KEY}")"
    FROZEN="$(frozen_for "${TASK}" "${MODEL_KEY}")"

    FROZEN_PATH="${PHASE2_ROOT}/outputs/frozen/${FROZEN}"
    # Hard fail: without the frozen JSONL, results would be incomplete and silently misleading.
    if [[ ! -f "${FROZEN_PATH}" ]]; then
      echo "[ERROR] Frozen file not found: ${FROZEN_PATH}"
      exit 1
    fi

    for L in "${LAYERS[@]}"; do
      TAG="${TASK}_${MODEL_KEY}_layer${L}"
      OUT_DIR="${OUT_ROOT}/${TASK}_${MODEL_KEY}/layer_${L}"
      mkdir -p "${OUT_DIR}"

      # Deterministic run configuration is entirely controlled via CLI flags passed through here.
      python "${RUN}" \
        --task "${TASK}" \
        --frozen_jsonl "${FROZEN_PATH}" \
        --out_jsonl "${OUT_DIR}/${TAG}.B${B}.results.jsonl" \
        --out_manifest "${OUT_DIR}/${TAG}.B${B}.manifest.json" \
        --model_name "${MODEL_NAME}" \
        --device "cuda:0" \
        --dtype "bfloat16" \
        --seed "${SEED}" \
        --n_splits "${N_SPLITS}" \
        --B "${B}" \
        --ci "${CI}" \
        --batch_size "${BS}" \
        --hidden_batch_size "${HBS}" \
        --hidden_layers "${L}" \
        --hidden_pooling "${HIDDEN_POOLING}" \
        --max_context_tokens "${MAX_CTX_TOK}"
    done

    echo "[OK] Hidden layer sweep finished for: ${TASK} × ${MODEL_KEY}. Outputs in: ${OUT_ROOT}/${TASK}_${MODEL_KEY}/"
  done
done

echo "[OK] Hidden layer sweep finished. Outputs root: ${OUT_ROOT}"