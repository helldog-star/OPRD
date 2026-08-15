#!/bin/bash
# Single-GPU (e.g. 1xH200) smoke for MOPD-OPRD (rep-only).
# Includes OOM mitigations from the 1xH200 pilot; not for formal 8xA100 comparison.
#
#   bash mopd_multi_teacher_oprd_1gpu.sh
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=True
export LOG_PROB_TOP_K=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_oprd_1gpu_$(date +%Y-%m-%d_%H-%M-%S)}
export MOPD_LOG_PREFIX=${MOPD_LOG_PREFIX:-mopd_oprd_1gpu}

export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-1024}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export N_GPUS=1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export N_RESPONSES=${N_RESPONSES:-4}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}

export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.25}
export RM_USE_DYNAMIC_BSZ=${RM_USE_DYNAMIC_BSZ:-False}
export RM_MICRO_BATCH_SIZE_PER_GPU=${RM_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}

export MIX_PARQUET=${MIX_PARQUET:-$SCRIPT_DIR/datasets/mopd_math_code_mix_8k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-mopd_math_code_mix_8k}

if [ ! -f "$MIX_PARQUET" ]; then
    GOPD_DATA_DIR=${GOPD_DATA_DIR:-${DATA_DIR:-/root/siton-tmp/home/liuxinyu/hf_datasets}/G-OPD-Training-Data}
    python "$SCRIPT_DIR/scripts/prepare_mopd_mix.py" \
        --math "$GOPD_DATA_DIR/DeepMath-103K/train_filtered_level6.parquet" \
        --code "$GOPD_DATA_DIR/Eurus/code_train.parquet" \
        --n_math 4000 --n_code 4000 \
        --out "$MIX_PARQUET"
fi

# Reuse 1gpu logits wrapper env + OPRD flags via main logits launcher.
exec bash "$SCRIPT_DIR/mopd_multi_teacher_logits.sh" "$@"
