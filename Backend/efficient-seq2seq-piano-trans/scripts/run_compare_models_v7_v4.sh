#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/Backend/efficient-seq2seq-piano-trans"
CHECKPOINT_PATH="${1:-${BACKEND_DIR}/checkpoints/T5_V4_steps_200000-OG.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
DEVICE="${DEVICE:-cuda}"
NUM_BATCHES="${NUM_BATCHES:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --backend-dir "${BACKEND_DIR}"
  --device "${DEVICE}"
  --num-batches "${NUM_BATCHES}"
  --warmup-runs "${WARMUP_RUNS}"
)

if [[ -n "${AUDIO_PATH:-}" ]]; then
  COMMON_ARGS+=(--audio-path "${AUDIO_PATH}")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--batch-size "${BATCH_SIZE}")
fi

python "${REPO_ROOT}/compare_models.py" \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config-name experiment_T5_V7_TurboQuantV2_FullDecoder \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config-name experiment_T5_V4_HierarchyPool \
  --output-csv "${OUTPUT_DIR}/compare_models_v7_vs_v4_same_checkpoint.csv"

python "${REPO_ROOT}/compare_models.py" \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config-name experiment_T5_V7_TurboQuantV2_FullDecoder \
  --output-csv "${OUTPUT_DIR}/compare_models_v7_single_same_checkpoint.csv"

python "${REPO_ROOT}/compare_models.py" \
  "${COMMON_ARGS[@]}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config-name experiment_T5_V4_HierarchyPool \
  --output-csv "${OUTPUT_DIR}/compare_models_v4_single_same_checkpoint.csv"
