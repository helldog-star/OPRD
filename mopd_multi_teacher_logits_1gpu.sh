#!/bin/bash
# Single-GPU (e.g. 1xH200) smoke for MOPD-logits.
# Short response / small batch; not for formal comparison with 8xA100 runs.
#
#   bash mopd_multi_teacher_logits_1gpu.sh
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export N_GPUS=1
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-4096}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-8}
export N_RESPONSES=${N_RESPONSES:-4}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-10000000000}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

export MIX_PARQUET=${MIX_PARQUET:-$SCRIPT_DIR/datasets/mopd_math_code_mix_8k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-mopd_math_code_mix_8k}
export MOPD_LOG_PREFIX=${MOPD_LOG_PREFIX:-mopd_logits_1gpu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_logits_1gpu_$(date +%Y-%m-%d_%H-%M-%S)}

# Optional: build a small mix if missing
if [ ! -f "$MIX_PARQUET" ]; then
    GOPD_DATA_DIR=${GOPD_DATA_DIR:-${DATA_DIR:-/root/siton-tmp/home/liuxinyu/hf_datasets}/G-OPD-Training-Data}
    python "$SCRIPT_DIR/scripts/prepare_mopd_mix.py" \
        --math "$GOPD_DATA_DIR/DeepMath-103K/train_filtered_level6.parquet" \
        --code "$GOPD_DATA_DIR/Eurus/code_train.parquet" \
        --n_math 4000 --n_code 4000 \
        --out "$MIX_PARQUET"
fi

exec bash "$SCRIPT_DIR/mopd_multi_teacher_logits.sh" "$@"
