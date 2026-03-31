#!/usr/bin/env bash
# scripts/run_hrm_rope_sweep.sh
#
# Three long-run sweeps for the TRM-style recursive encoder. The focus is now
# on halting-target sharpness, halt-loss weight, and recursion depth.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LONG_STEPS="${LONG_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"

RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_C="${RUN_C:-1}"

echo "======================================================="
echo "TRM RoPE sweep — $(date)"
echo "ROOT_DIR  = $ROOT_DIR"
echo "LONG_STEPS= $LONG_STEPS   EVAL_EVERY= $EVAL_EVERY"
echo "Runs:  A=${RUN_A}  B=${RUN_B}  C=${RUN_C}"
echo "======================================================="

run_case() {
  local name="$1"
  shift
  echo ""
  echo "========== RUN: ${name} =========="
  python train.py "$@"
}

# Run A — calibration baseline.
if [[ "${RUN_A}" == "1" ]]; then
  run_case "trm_rope_calib_sharp4_weight010_steps4" \
    training.notes="TRM_rope_calib_sharp4_weight010_steps4" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.trm_halt_target_sharpness=4.0 \
    training.trm_halt_loss_weight=0.10 \
    model.trm_recursions=4 \
    model.trm_min_recursions=2 \
    model.trm_halt_threshold=0.5
fi

# Run B — smoother target with deeper recursion.
if [[ "${RUN_B}" == "1" ]]; then
  run_case "trm_rope_soft_sharp2_weight015_steps6" \
    training.notes="TRM_rope_soft_sharp2_weight015_steps6" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.trm_halt_target_sharpness=2.0 \
    training.trm_halt_loss_weight=0.15 \
    model.trm_recursions=6 \
    model.trm_min_recursions=2 \
    model.trm_halt_threshold=0.5
fi

# Run C — harder target as an ablation.
if [[ "${RUN_C}" == "1" ]]; then
  run_case "trm_rope_hard_sharp8_weight010_steps4" \
    training.notes="TRM_rope_hard_sharp8_weight010_steps4" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.trm_halt_target_sharpness=8.0 \
    training.trm_halt_loss_weight=0.10 \
    model.trm_recursions=4 \
    model.trm_min_recursions=2 \
    model.trm_halt_threshold=0.5
fi

echo ""
echo "======================================================="
echo "All requested runs complete — $(date)"
echo ""
echo "Key metrics to compare in WandB (val/):"
echo "  note_f1               — primary transcription accuracy"
echo "  note_with_offset_f1   — stricter; checks onset timing"
echo "  trm_predicted_steps_mean"
echo "                         — mean recursion count implied by the halt head"
echo "  trm_halt_logit_mean   — final-step halting confidence"
echo "  trm_halt_to_ce_ratio  — halt-loss fraction of total loss"
echo ""
echo "Promotion heuristic: pick the run with the highest note_with_offset_f1"
echo "and stable trm_predicted_steps_mean below the maximum recursion budget."
echo ""
echo "Example promoted long run (adjust notes/steps as needed):"
echo "  python train.py \\"
echo "    training.notes=TRM_rope_promoted_long \\"
echo "    training.training_steps=100000 \\"
echo "    training.evaluate_every_n_steps=1000 \\"
echo "    model.trm_halt_target_sharpness=<winner_sharpness> \\"
echo "    training.trm_halt_loss_weight=<winner_weight> \\"
echo "    model.trm_recursions=<winner_steps>"
echo "======================================================="
