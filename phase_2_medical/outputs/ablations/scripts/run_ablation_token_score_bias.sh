#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------
# Script: Token Score Bias ablation runner (TruthfulQA; LNTP/MTP + answer-span k-sweep)
# - Purpose: Invoke the phase_2_medical ablation runner with fixed, reproducible settings.
# - Key input: frozen TruthfulQA JSONL (precomputed prompts/metadata) + optional HF auth (.env).
# - Key outputs: per-model JSONL results + manifests under outputs/ablations/token_score_bias/<model_key>/.
# - Determinism: deterministic at this wrapper level (fixed seed, fixed params); full determinism also depends
#   on the Python runner and GPU kernel determinism settings.
# --------------------------------------------------

# Load HF token from .env (if available) so model downloads/authenticated pulls work without manual export.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "[INFO] Loading HF token from .env"
  set -a
  source "${REPO_ROOT}/.env"
  set +a

  # Convention: accept either HF_TOKEN or HUGGINGFACE_HUB_TOKEN; export the latter for huggingface_hub tooling.
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
  fi
else
  # NOTE: potential issue: unauthenticated pulls may hit rate limits or fail for gated models.
  echo "[INFO] No .env found at repo root (continuing without explicit HF token)"
fi


# Ablation: Token Score Bias (LNTP/MTP)
# Includes BOTH:
#   (1) Length Normalization: mean vs sum (+ score-length Spearman)
#   (2) Answer-Span Ablation: k-sweep on first k answer tokens
#
# Outputs:
#   phase_2_medical/outputs/ablations/token_score_bias/truthfulqa_<model_key>/
#     - token_bias.results.jsonl
#     - token_bias.manifest.json              (includes both ablations)
#     - k_sweep.manifest.json                 (optional, only for convenience)

# Robust path resolution (independent of where the script is invoked from).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"     # -> phase_2_medical

# Runner + I/O: keep paths explicit to avoid accidental cross-phase writes.
RUN="${PHASE2_ROOT}/src/run_token_bias_lntp_mtp.py"
OUT_ROOT="${PHASE2_ROOT}/outputs/ablations/token_score_bias"
FROZEN="${PHASE2_ROOT}/outputs/frozen/truthfulqa_hallu_mistral_like.jsonl"

mkdir -p "${OUT_ROOT}"

# Heuristic: reduce CUDA allocator fragmentation for long-running generation/scoring workloads.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Fixed params (kept centralized to make runs comparable across models).
SEED=42
DEVICE="cuda:0"
DTYPE="bfloat16"
BATCH_SIZE=8
MAX_INPUT_TOKENS=512

# Answer-span sweep (Ablation 2): evaluates sensitivity of scores to truncating the answer to first k tokens.
K_LIST=(1 2 3 5 10 20)

# Models: keys are stable identifiers used for output folder naming and manifest provenance.
MODEL_KEYS=("mistral")

model_name_for_key () {
  # Mapping from stable model key -> HF model identifier used by the Python runner.
  case "$1" in
    mistral)    echo "mistralai/Mistral-7B-Instruct-v0.2" ;;
    biomistral) echo "BioMistral/BioMistral-7B" ;;
    # TODO: verify: add new model keys here before enabling them in MODEL_KEYS to avoid "UNKNOWN" failures.
    *) echo "UNKNOWN" ; exit 1 ;;
  esac
}

# Sanity checks: fail fast to avoid partial runs producing misleadingly incomplete artifacts.
[[ -f "${RUN}" ]] || { echo "[ERROR] Runner not found: ${RUN}"; exit 1; }
[[ -f "${FROZEN}" ]] || { echo "[ERROR] Frozen file not found: ${FROZEN}"; exit 1; }

echo "[INFO] PHASE2_ROOT=${PHASE2_ROOT}"
echo "[INFO] Frozen:     ${FROZEN}"
echo "[INFO] Out root:   ${OUT_ROOT}"
echo "[INFO] K_LIST:     ${K_LIST[*]}"
echo

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  MODEL_NAME="$(model_name_for_key "${MODEL_KEY}")"
  OUT_DIR="${OUT_ROOT}/${MODEL_KEY}"
  mkdir -p "${OUT_DIR}"

  echo "=== Token Score Bias: TruthfulQA × ${MODEL_KEY} ==="

  # NOTE: potential issue: results overwrite is intentional (fixed filenames); keep OUT_DIR unique per model_key.
  python "${RUN}" \
    --frozen_jsonl "${FROZEN}" \
    --out_jsonl "${OUT_DIR}/token_bias.results.jsonl" \
    --out_manifest "${OUT_DIR}/token_bias.manifest.json" \
    --out_k_sweep_manifest "${OUT_DIR}/k_sweep.manifest.json" \
    --model_name "${MODEL_NAME}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --seed "${SEED}" \
    --batch_size "${BATCH_SIZE}" \
    --max_input_tokens "${MAX_INPUT_TOKENS}" \
    --k_list "${K_LIST[@]}"

  echo
done

echo "[OK] Token Score Bias ablation finished. Outputs root: ${OUT_ROOT}"