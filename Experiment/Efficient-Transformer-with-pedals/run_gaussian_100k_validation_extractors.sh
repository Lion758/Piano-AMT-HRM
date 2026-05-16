#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT_PATH="/home/rachel/.group-5/Piano-AMT-HRM/Experiment/Efficient-Transformer-with-pedals/runs/Transformer-T5/260515-043822_Efficient_Transformer_V4_ConvHeads_140k_gaussian_/AMT-audio-to-midi/pnahxhsn/checkpoints/epoch=79-step=140000.ckpt"
BASE_EVAL_DIR="evaluations/Gaussian_QFL_140k_validation"
BATCH_TEST=100
LOG_DIR="evaluations/logs"

mkdir -p "${LOG_DIR}"

run_eval() {
  local extractor="$1"
  local output_dir="${BASE_EVAL_DIR}_${extractor}"
  local log_path="${LOG_DIR}/Gaussian_QFL_100k_validation_${extractor}.log"

  echo "Running validation evaluation with ${extractor} extractor..."
  echo "Summary CSV will be written to: ${output_dir}/!test_metrics_summary.csv"
  python evaluate.py --config-name evaluate_maestro_temporal_conv_heads_validation \
    model.checkpoint_path="'${CHECKPOINT_PATH}'" \
    model.mlp_activations=relu \
    model.strict_checkpoint=false \
    evaluation.subset=validation \
    evaluation.pedal_event_source=frame_head \
    evaluation.midi_pedal_event_source=frame_head \
    evaluation.frame_head_event_extractor="${extractor}" \
    evaluation.save_track_json=false \
    evaluation.save_output_midi=false \
    evaluation.copy_reference_midi=false \
    evaluation.test_output_dir="${output_dir}" \
    training.batch_test="${BATCH_TEST}" \
    training.notes="Gaussian_QFL_100k_validation_${extractor}" 2>&1 | tee "${log_path}"
}

run_eval state_hysteresis
run_eval trend_dual_trigger

echo "Done."
echo "State hysteresis results: ${BASE_EVAL_DIR}_state_hysteresis"
echo "Trend dual trigger results: ${BASE_EVAL_DIR}_trend_dual_trigger"
