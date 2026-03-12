#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHORT_STEPS="${SHORT_STEPS:-3000}"
LONG_STEPS="${LONG_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-200}"

echo "HRM low-cost sweep"
echo "ROOT_DIR=$ROOT_DIR"
echo "SHORT_STEPS=$SHORT_STEPS LONG_STEPS=$LONG_STEPS EVAL_EVERY=$EVAL_EVERY"

run_case() {
  local name="$1"
  shift
  echo ""
  echo "========== RUN: ${name} =========="
  python train.py "$@"
}

# 1) Gate run baseline: current V5 setup + constant Q-loss weight.
run_case "gate_const_q005_steps4" \
  training.notes="HRM_gate_const_q005_steps4" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.hrm_q_loss_weight=0.05 \
  training.hrm_q_loss_init_weight=0.05 \
  training.hrm_q_loss_warmup_steps=0 \
  training.hrm_q_loss_ramp_steps=0 \
  model.hrm_max_steps=4

# 2) Q-loss schedule ablation: warmup + ramp to 0.05.
run_case "ablation_qramp_to_q005_steps4" \
  training.notes="HRM_ablation_qramp_steps4" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.hrm_q_loss_weight=0.05 \
  training.hrm_q_loss_init_weight=0.0 \
  training.hrm_q_loss_warmup_steps=1000 \
  training.hrm_q_loss_ramp_steps=4000 \
  model.hrm_max_steps=4

# 3) ACT-depth ablation: steps 6 (with the same ramp schedule).
run_case "ablation_qramp_to_q005_steps6" \
  training.notes="HRM_ablation_qramp_steps6" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.hrm_q_loss_weight=0.05 \
  training.hrm_q_loss_init_weight=0.0 \
  training.hrm_q_loss_warmup_steps=1000 \
  training.hrm_q_loss_ramp_steps=4000 \
  model.hrm_max_steps=6

echo ""
echo "Select the best short run by validation note_f1 with no major note+offset_f1 drop."
echo "Then launch one longer follow-up run, for example:"
echo "python train.py training.notes=HRM_promoted_long training.training_steps=${LONG_STEPS} training.hrm_q_loss_weight=0.05 training.hrm_q_loss_init_weight=0.0 training.hrm_q_loss_warmup_steps=1000 training.hrm_q_loss_ramp_steps=4000 model.hrm_max_steps=4"
