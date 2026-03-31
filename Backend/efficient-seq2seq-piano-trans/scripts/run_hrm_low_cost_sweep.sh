#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHORT_STEPS="${SHORT_STEPS:-3000}"
LONG_STEPS="${LONG_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-200}"

echo "TRM low-cost sweep"
echo "ROOT_DIR=$ROOT_DIR"
echo "SHORT_STEPS=$SHORT_STEPS LONG_STEPS=$LONG_STEPS EVAL_EVERY=$EVAL_EVERY"

run_case() {
  local name="$1"
  shift
  echo ""
  echo "========== RUN: ${name} =========="
  python train.py "$@"
}

# 1) Conservative baseline: 4 recursions, standard halt loss weight.
run_case "trm_baseline_weight005_steps4" \
  training.notes="TRM_baseline_weight005_steps4" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.trm_halt_loss_weight=0.05 \
  model.trm_recursions=4

# 2) Heavier halting supervision at the same depth.
run_case "trm_weight010_steps4" \
  training.notes="TRM_weight010_steps4" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.trm_halt_loss_weight=0.10 \
  model.trm_recursions=4

# 3) Deeper recursion with the baseline halting weight.
run_case "trm_baseline_weight005_steps6" \
  training.notes="TRM_baseline_weight005_steps6" \
  training.training_steps="$SHORT_STEPS" \
  training.evaluate_every_n_steps="$EVAL_EVERY" \
  training.trm_halt_loss_weight=0.05 \
  model.trm_recursions=6

echo ""
echo "Select the best short run by validation note_f1 with no major note+offset_f1 drop."
echo "Then launch one longer follow-up run, for example:"
echo "python train.py training.notes=TRM_promoted_long training.training_steps=${LONG_STEPS} training.trm_halt_loss_weight=0.05 model.trm_recursions=6"
