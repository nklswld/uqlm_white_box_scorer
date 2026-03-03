#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# OOF seed ablation runner (Phase 2, medical tasks)
#
# Runs out-of-fold (OOF) robustness experiments by varying only --seed while
# keeping frozen inputs, model choice, bootstrap settings (B, ci), and CV
# settings (n_splits) fixed. Reads optional HF credentials from repo-root .env.
#
# Key inputs: frozen JSONL under outputs/frozen/, TASKS, MODEL_KEYS, SEEDS, and
# run_phase2.py. Key outputs: per-seed results/manifest under outputs/ablations/
# oof_seeds/<task>_<model_key>/seed_<seed>/. Determinism: controlled via --seed
# and fixed frozen inputs; assumes downstream code is seed-respecting.
# --------------------------------------------------

# --------------------------------------------------
# Load HF token from .env (if available)
# --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "[INFO] Loading HF token from .env"
  set -a
  source "${REPO_ROOT}/.env"
  set +a

  # Accept both conventions: HF_TOKEN (repo) and HUGGINGFACE_HUB_TOKEN (HF SDK).
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
else
  # NOTE: potential issue: private/gated HF models will fail without a valid token.
  echo "[INFO] No .env found at repo root (continuing without explicit HF token)"
fi


# Ablation: OOF Robustness – Seeds
# Vary: --seed
# Fixed: everything else (including frozen inputs, model, B, n_splits, hidden settings)
#
# Output: ablations/oof_seeds/<task>_<model_key>/seed_<seed>/

# Robust path resolution (independent of where the script is invoked from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"           # -> phase_2_medical
REPO_ROOT="$(cd "${PHASE2_ROOT}/.." && pwd)"           # -> repo root

RUN="${PHASE2_ROOT}/src/run_phase2.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/oof_seeds"
mkdir -p "${OUT_ROOT}"

# Optional debug info for reproducible runs/log provenance.
echo "[INFO] PHASE2_ROOT=${PHASE2_ROOT}"
echo "[INFO] Using frozen dir: ${PHASE2_ROOT}/outputs/frozen"

B=5000
CI=0.95
N_SPLITS=5

# Hidden-state extraction settings forwarded to run_phase2.py.
HIDDEN_LAYERS="16"
HIDDEN_POOLING="mean_answer"

# Avoid CUDA allocator fragmentation OOMs for long-running batched inference.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Tasks
TASKS=("medqa" "pubmedqa")

# Model keys
MODEL_KEYS=("mistral" "biomistral")

# Seeds to test; expanded to probe sensitivity rather than maximize coverage.
SEEDS=(0 42 123 999 2026)

# Map short model keys to canonical HF model IDs (single source of truth).
model_name_for_key () {
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Resolve the exact frozen JSONL name per (task, model_key).
# Invariant: these files must be identical across seeds for a fair ablation.
frozen_for () {
  case "$1-$2" in
    medqa-mistral)       echo "medqa_mistral7b.jsonl" ;;
    medqa-biomistral)    echo "medqa_biomistral7b.jsonl" ;;
    pubmedqa-mistral)    echo "pubmedqa_mistral7b.jsonl" ;;
    pubmedqa-biomistral) echo "pubmedqa_biomistral7b.jsonl" ;;
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Task-specific runtime defaults, aligned with prior manual runs for comparability.
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

for TASK in "${TASKS[@]}"; do
  read -r BS HBS MAX_CTX_TOK <<<"$(task_params "${TASK}")"

  for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL_NAME="$(model_name_for_key "${MODEL_KEY}")"
    FROZEN="$(frozen_for "${TASK}" "${MODEL_KEY}")"

    FROZEN_PATH="${PHASE2_ROOT}/outputs/frozen/${FROZEN}"

    # Fail fast: missing frozen inputs would silently invalidate the ablation.
    if [[ ! -f "${FROZEN_PATH}" ]]; then
      echo "[ERROR] Frozen file not found: ${FROZEN_PATH}"
      exit 1
    fi

    for SEED in "${SEEDS[@]}"; do
      TAG="${TASK}_${MODEL_KEY}_seed${SEED}"
      OUT_DIR="${OUT_ROOT}/${TASK}_${MODEL_KEY}/seed_${SEED}"
      mkdir -p "${OUT_DIR}"

      # Invariant: only --seed changes across runs; all other knobs are fixed.
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
        --hidden_layers ${HIDDEN_LAYERS} \
        --hidden_pooling "${HIDDEN_POOLING}" \
        --max_context_tokens "${MAX_CTX_TOK}"
    done

    echo "[OK] Finished: ${TASK} × ${MODEL_KEY} (seeds: ${SEEDS[*]}). Outputs in: ${OUT_ROOT}/${TASK}_${MODEL_KEY}/"
  done
done

echo "[OK] OOF seed ablation finished. Outputs root: ${OUT_ROOT}"