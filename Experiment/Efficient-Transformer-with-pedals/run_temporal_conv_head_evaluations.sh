#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TEST_EVAL_DIR="evaluations/MAESTRO_TemporalConvHeads_30k_test"
VALIDATION_EVAL_DIR="evaluations/MAESTRO_TemporalConvHeads_30k_validation"
VALIDATION_SWEEP_CSV="${VALIDATION_EVAL_DIR}/threshold_sweep_validation_wide.csv"

echo "Running MAESTRO test evaluation for temporal conv pedal heads..."
python evaluate.py --config-name evaluate_maestro_temporal_conv_heads_test \
  evaluation.test_output_dir="${TEST_EVAL_DIR}"

echo "Running MAESTRO validation evaluation for temporal conv pedal heads..."
python evaluate.py --config-name evaluate_maestro_temporal_conv_heads_validation \
  evaluation.test_output_dir="${VALIDATION_EVAL_DIR}"

echo "Running validation threshold sweep from saved frame-head JSON..."
python tools/sweep_pedal_frame_thresholds.py \
  --eval-dir "${VALIDATION_EVAL_DIR}" \
  --output-csv "${VALIDATION_SWEEP_CSV}" \
  --threshold-on-values 0.30,0.35,0.40,0.45,0.50,0.55,0.60 \
  --threshold-off-values 0.20,0.25,0.30,0.35,0.40,0.45 \
  --min-down-frames 1,2,3,4,5,6 \
  --min-up-frames 1,2,3,4,5,6

echo "Done. Validation sweep CSV: ${VALIDATION_SWEEP_CSV}"
