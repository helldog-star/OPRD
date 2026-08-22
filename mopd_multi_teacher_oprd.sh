#!/bin/bash
# MOPD-OPRD: same multi-teacher routing as mopd_multi_teacher_logits.sh,
# but teacher signal is representation MSE (rep-only) instead of token reverse-KL.
#
# Keeps batch / n / response length identical to the logits arm for fair comparison.
# 8x A100 only. Single-GPU: bash mopd_multi_teacher_oprd_1gpu.sh
#
#   bash mopd_multi_teacher_oprd.sh
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=True
export LOG_PROB_TOP_K=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_oprd_rep_$(date +%Y-%m-%d_%H-%M-%S)}
export MOPD_LOG_PREFIX=${MOPD_LOG_PREFIX:-mopd_oprd}

# Official OPRD rep defaults (override via env if needed)
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}

export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}
export RM_USE_DYNAMIC_BSZ=${RM_USE_DYNAMIC_BSZ:-True}
export RM_MICRO_BATCH_SIZE_PER_GPU=${RM_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
# Keep models on GPU. Teacher hidden is scored per ppo mini-batch (bf16) and dropped.
# Offload + full-step fp32 hidden cache is what blew CPU RAM previously.
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

exec bash "$SCRIPT_DIR/mopd_multi_teacher_logits.sh" "$@"
