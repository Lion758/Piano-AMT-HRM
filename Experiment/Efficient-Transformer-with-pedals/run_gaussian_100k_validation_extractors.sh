#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CHECKPOINT_PATH="/home/rachel/.group-5/Piano-AMT-HRM/Experiment/Efficient-Transformer-with-pedals/runs/Transformer-T5/260514-154114_Efficient_Transformer_V4_ConvHeads_100k_gaussian_/AMT-audio-to-midi/cuim96l1/checkpoints/epoch=56-step=100000.ckpt"
BASE_EVAL_DIR="evaluations/Gaussian_QFL_100k_validation"
BATCH_TEST=150

echo "Running validation evaluation with state_hysteresis extractor..."
python evaluate.py --config-name evaluate_maestro_temporal_conv_heads_validation \
  model.checkpoint_path="'${CHECKPOINT_PATH}'" \
  model.mlp_activations=relu \
  model.strict_checkpoint=false \
  evaluation.subset=validation \
  evaluation.pedal_event_source=frame_head \
  evaluation.midi_pedal_event_source=frame_head \
  evaluation.frame_head_event_extractor=state_hysteresis \
  evaluation.test_output_dir="${BASE_EVAL_DIR}_state_hysteresis" \
  training.batch_test="${BATCH_TEST}" \
  training.notes="Gaussian_QFL_100k_validation_state_hysteresis"

echo "Running validation evaluation with trend_dual_trigger extractor..."
python evaluate.py --config-name evaluate_maestro_temporal_conv_heads_validation \
  model.checkpoint_path="'${CHECKPOINT_PATH}'" \
  model.mlp_activations=relu \
  model.strict_checkpoint=false \
  evaluation.subset=validation \
  evaluation.pedal_event_source=frame_head \
  evaluation.midi_pedal_event_source=frame_head \
  evaluation.frame_head_event_extractor=trend_dual_trigger \
  evaluation.test_output_dir="${BASE_EVAL_DIR}_trend_dual_trigger" \
  training.batch_test="${BATCH_TEST}" \
  training.notes="Gaussian_QFL_100k_validation_trend_dual_trigger"

echo "Done."
echo "State hysteresis results: ${BASE_EVAL_DIR}_state_hysteresis"
echo "Trend dual trigger results: ${BASE_EVAL_DIR}_trend_dual_trigger"
