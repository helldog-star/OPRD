#!/bin/bash
# MOPD-OPRD pilot: same multi-teacher routing as mopd_multi_teacher_logits.sh,
# but teacher signal is representation MSE (rep-only) instead of token reverse-KL.
#
#   bash mopd_multi_teacher_oprd.sh
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=True
export LOG_PROB_TOP_K=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_oprd_rep_$(date +%Y-%m-%d_%H-%M-%S)}
# Reuse logits launcher; override rep knobs via env before exec.
exec bash "$SCRIPT_DIR/mopd_multi_teacher_logits.sh" "$@"
