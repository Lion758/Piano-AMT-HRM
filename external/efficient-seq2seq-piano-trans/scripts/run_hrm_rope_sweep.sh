#!/usr/bin/env bash
# scripts/run_hrm_rope_sweep.sh
#
# Three long-run sweeps for the RoPE + Q-loss improvements introduced in
# hrm_encoder.py.  Run sequentially by default; see PARALLEL note below.
#
# Key variables being swept (all new or changed by the refactor):
#
#   hrm_q_soft_k          — sharpness of soft is_easy target.
#                           Low  (2.0) = smooth gradient everywhere, forgiving
#                           High (20.0) = near-hard binary (ablation baseline)
#                           Default (5.0) = moderate, boundary-aware
#
#   hrm_q_loss_weight     — total Q-loss scale.  Raised from old 0.05 default
#                           because switching from reduction='sum' to 'mean'
#                           shrinks the raw loss value by ~batch_size.
#
#   hrm_max_steps         — ACT depth (refinement iterations per encoder call).
#                           More steps = deeper reasoning, more compute.
#
#   hrm_halt_explore_prob — probability of forcing a random minimum step count
#                           before allowing early halt.  Higher = more uniform
#                           step distribution during training = more Q-head data.
#
#   hrm_q_loss_warmup_steps / hrm_q_loss_ramp_steps — schedule for Q-loss weight
#                           (already supported by train.py _get_hrm_q_loss_weight).
#                           Longer ramp needed when max_steps is larger.
#
# Usage:
#   bash scripts/run_hrm_rope_sweep.sh
#
# Override step counts without editing this file:
#   LONG_STEPS=50000 EVAL_EVERY=500 bash scripts/run_hrm_rope_sweep.sh
#
# To run a single case:
#   RUN_A=1 RUN_B=0 RUN_C=0 bash scripts/run_hrm_rope_sweep.sh
#
# PARALLEL: to run all three simultaneously on separate GPUs, set
#   CUDA_VISIBLE_DEVICES=0,1,2 and launch each block in background (&).
#   This script runs them sequentially to keep GPU memory simple.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ── Step-count knobs ──────────────────────────────────────────────────────────
LONG_STEPS="${LONG_STEPS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-500}"

# ── Run selector (set to 0 to skip a run) ────────────────────────────────────
RUN_A="${RUN_A:-1}"
RUN_B="${RUN_B:-1}"
RUN_C="${RUN_C:-1}"

echo "======================================================="
echo "HRM RoPE sweep — $(date)"
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


# ─────────────────────────────────────────────────────────────────────────────
# Run A — "rope_calib"
#
# Calibration run: moderate soft-k (close to the hard binary but with gradient
# at the boundary), doubled Q-loss weight to compensate for reduction='mean',
# and the same ACT depth (4 steps) as the old V5 baseline.
#
# Use this as the reference point: if note_f1 is similar to the old V5 short
# runs but hrm_steps_mean converges toward 2–3 (not always 4), the Q-head is
# learning to halt early on easy passages.
#
# Sweep focus:
#   hrm_q_soft_k      = 5.0   (new default; moderate boundary smoothing)
#   hrm_q_loss_weight = 0.10  (2× old 0.05, compensates for mean vs sum)
#   hrm_max_steps     = 4
#   hrm_halt_explore_prob = 0.10
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_A}" == "1" ]]; then
  run_case "rope_calib_softk5_q010_steps4" \
    training.notes="HRM_rope_calib_softk5_q010_steps4" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.hrm_q_soft_k=5.0 \
    training.hrm_q_loss_weight=0.10 \
    training.hrm_q_loss_init_weight=0.0 \
    training.hrm_q_loss_warmup_steps=1000 \
    training.hrm_q_loss_ramp_steps=4000 \
    model.hrm_max_steps=5 \
    model.hrm_halt_explore_prob=0.10
fi


# ─────────────────────────────────────────────────────────────────────────────
# Run B — "rope_soft_deep"
#
# Softer Q-target (k=2) combined with deeper ACT (6 steps).
#
# k=2 gives a wide gradient signal around the loss median — useful when the
# batch loss distribution is tight (early training, similar difficulty examples)
# so the binary threshold would fire randomly.  Softer targets let the Q-head
# learn a more reliable difficulty ranking before committing to halt/continue.
#
# 6 ACT steps increases encoder refinement capacity at a ~50% compute cost.
# The Q-loss ramp is extended to give the deeper model time to warm up before
# the halt penalty kicks in (otherwise it learns to always use 1 step to avoid
# the loss, before the decoder has learned anything useful).
#
# Sweep focus:
#   hrm_q_soft_k      = 2.0   (soft/smooth; gradient at the boundary)
#   hrm_q_loss_weight = 0.15  (slightly higher for softer targets)
#   hrm_max_steps     = 6     (deeper reasoning, more compute)
#   hrm_halt_explore_prob = 0.10
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_B}" == "1" ]]; then
  run_case "rope_soft_softk2_q015_steps6" \
    training.notes="HRM_rope_soft_softk2_q015_steps6" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.hrm_q_soft_k=2.0 \
    training.hrm_q_loss_weight=0.15 \
    training.hrm_q_loss_init_weight=0.0 \
    training.hrm_q_loss_warmup_steps=2000 \
    training.hrm_q_loss_ramp_steps=8000 \
    model.hrm_max_steps=4 \
    model.hrm_halt_explore_prob=0.10
fi


# ─────────────────────────────────────────────────────────────────────────────
# Run C — "rope_hard_explore"
#
# Near-hard binary target (k=20, effectively the old hard is_easy) combined
# with higher exploration probability.
#
# This serves as an ablation on soft_k: if Run A/B don't beat Run C, the
# smoother target isn't helping and the hard binary was fine.
#
# Higher halt_explore_prob (0.20 vs 0.10) forces more varied step counts during
# training, giving the Q-head a richer signal about what each extra step buys.
# This is especially important with a hard target: if the model almost always
# halts at step 1 (easy passages dominate MAESTRO), exploration ensures it sees
# what happens at steps 2–4 often enough to learn to use them when needed.
#
# Sweep focus:
#   hrm_q_soft_k           = 20.0  (near-hard binary; ablation of soft target)
#   hrm_q_loss_weight      = 0.10
#   hrm_max_steps          = 4
#   hrm_halt_explore_prob  = 0.20  (double exploration for richer Q-head signal)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${RUN_C}" == "1" ]]; then
  run_case "rope_hard_explore_softk20_q010_steps4_expl02" \
    training.notes="HRM_rope_hard_explore_softk20_q010_steps4_expl02" \
    training.training_steps="$LONG_STEPS" \
    training.evaluate_every_n_steps="$EVAL_EVERY" \
    model.hrm_q_soft_k=20.0 \
    training.hrm_q_loss_weight=0.10 \
    training.hrm_q_loss_init_weight=0.0 \
    training.hrm_q_loss_warmup_steps=1000 \
    training.hrm_q_loss_ramp_steps=4000 \
    model.hrm_max_steps=4 \
    model.hrm_halt_explore_prob=0.20
fi


echo ""
echo "======================================================="
echo "All requested runs complete — $(date)"
echo ""
echo "Key metrics to compare in WandB (val/):"
echo "  note_f1              — primary transcription accuracy"
echo "  note_with_offset_f1  — stricter; checks onset timing"
echo "  hrm_steps_mean       — mean ACT steps used (should trend < max_steps)"
echo "  hrm_q_halt_logit_mean / hrm_q_continue_logit_mean"
echo "                       — Q-head confidence; diverging = Q-head is active"
echo "  hrm_q_to_ce_ratio    — Q-loss fraction of total loss (watch for dominance)"
echo ""
echo "Promotion heuristic: pick the run with the highest note_with_offset_f1"
echo "and hrm_steps_mean settling below max_steps - 1 by step 10k."
echo ""
echo "Example promoted long run (adjust notes/steps as needed):"
echo "  python train.py \\"
echo "    training.notes=HRM_rope_promoted_long \\"
echo "    training.training_steps=100000 \\"
echo "    training.evaluate_every_n_steps=1000 \\"
echo "    model.hrm_q_soft_k=<winner_k> \\"
echo "    training.hrm_q_loss_weight=<winner_weight> \\"
echo "    model.hrm_max_steps=<winner_steps> \\"
echo "    model.hrm_halt_explore_prob=<winner_explore>"
echo "======================================================="
