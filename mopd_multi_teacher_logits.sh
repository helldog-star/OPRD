#!/bin/bash
# MOPD-logits: multi-teacher on-policy distillation (sample-level routing by ability).
# Student = Qwen3-4B; teachers = Math/Code Non-Thinking RL Step1200.
#
# 8x A100 formal run. Hyperparams follow G-OPD same-size distillation
# (batch=1024, n=1, prompt=2048, resp=16384, lr=1e-5) and MOPD's N=1 design.
# Paper refs:
#   - MOPD: https://arxiv.org/html/2606.30406v1  (BS=2048, N=1 at 30B scale)
#   - G-OPD: https://arxiv.org/abs/2602.12125   (Table 6: BS=1024, N=1, resp=16384)
#
#   bash mopd_multi_teacher_logits.sh
# Single-GPU smoke: bash mopd_multi_teacher_logits_1gpu.sh
set -eo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- host runtime ---
export OPRD_CONDA_SH=${OPRD_CONDA_SH:-/root/siton-tmp/home/liuxinyu/miniconda3/etc/profile.d/conda.sh}
export OPRD_CONDA_ENV=${OPRD_CONDA_ENV:-verl}
export OPRD_CONDA_BIN=${OPRD_CONDA_BIN:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin}
# shellcheck disable=SC1090
source "$OPRD_CONDA_SH"
conda activate "$OPRD_CONDA_ENV"
export PATH="$OPRD_CONDA_BIN:$PATH"
export PYTHONPATH="${SCRIPT_DIR}/verl:${PYTHONPATH:-}"

# Clear proxies for Ray (socks ALL_PROXY / Docker IP hangs are common on siton hosts).
export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1,172.17.0.4,172.17.0.0/16}
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy=

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export WANDB_MODE=${WANDB_MODE:-offline}

export MODEL_DIR=${MODEL_DIR:-/root/siton-tmp/home/liuxinyu/hf_models}
export DATA_DIR=${DATA_DIR:-/root/siton-tmp/home/liuxinyu/hf_datasets}
export GOPD_DATA_DIR=${GOPD_DATA_DIR:-$DATA_DIR/G-OPD-Training-Data}
export PROJECT_PATH=${PROJECT_PATH:-./outputs}
export PROJECT_NAME=${PROJECT_NAME:-MOPD_MultiTeacher}

# --- algorithm ---
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}          # 0 = full-vocab reverse-KL (MOPD PG form)
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
export REP_DISTILLATION_ONLY=${REP_DISTILLATION_ONLY:-False}

# 8x A100 defaults (G-OPD Table 6 + MOPD N=1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export N_GPUS=${N_GPUS:-8}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
# Unified cap: match math teacher GRPO / G-OPD distill (code teacher was 8192 in RL;
# distillation used 16384 for both domains in G-OPD Table 6).
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-16384}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-16384}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}   # prompts / step (G-OPD distill)
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}       # ppo mini-batch
export N_RESPONSES=${N_RESPONSES:-1}                 # MOPD / G-OPD distill use n=1
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.55}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
export RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-80000000000}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}

export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-20}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export ACTOR_LR=${ACTOR_LR:-1e-5}

# Memory knobs (OPRD wrapper may override)
export RM_USE_DYNAMIC_BSZ=${RM_USE_DYNAMIC_BSZ:-True}
export RM_MICRO_BATCH_SIZE_PER_GPU=${RM_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
export MOPD_LOG_PREFIX=${MOPD_LOG_PREFIX:-mopd_logits}

export RAY_PORT=${RAY_PORT:-6399}
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_${RAY_PORT}}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export SKIP_RAY_STOP=${SKIP_RAY_STOP:-0}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    # Ensure prefix is set before creating the log file name.
    export MOPD_LOG_PREFIX=${MOPD_LOG_PREFIX:-mopd_logits}
    LOG_FILE="${LOG_DIR}/${MOPD_LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log file: $LOG_FILE"
fi

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-$MODEL_DIR/Qwen3-4B}
export MATH_TEACHER_PATH=${MATH_TEACHER_PATH:-$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step1200}
export CODE_TEACHER_PATH=${CODE_TEACHER_PATH:-$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Code-Step1200}

# Default: 0.5/0.5 math:code balanced mix (see prepare_mopd_mix.py --balance).
MIX_PARQUET=${MIX_PARQUET:-$SCRIPT_DIR/datasets/mopd_math_code_mix_balanced.parquet}
if [ ! -f "$MIX_PARQUET" ]; then
    echo "[prep] building balanced mix $MIX_PARQUET from G-OPD data"
    python "$SCRIPT_DIR/scripts/prepare_mopd_mix.py" \
        --math "$GOPD_DATA_DIR/DeepMath-103K/train_filtered_level6.parquet" \
        --code "$GOPD_DATA_DIR/Eurus/code_train.parquet" \
        --balance --math_ratio 0.5 \
        --out "$MIX_PARQUET"
fi
export TRAIN_DATASET="$MIX_PARQUET"
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-mopd_math_code_mix_balanced}

export TEST_DATA_DIR=${TEST_DATA_DIR:-$SCRIPT_DIR/datasets/test_data}
TEST_DATASET=${TEST_FILE:-$TEST_DATA_DIR/mopd_val_mix.parquet}
echo "[mopd] n_gpus=$N_GPUS train_batch=$TRAIN_BATCH_SIZE n=$N_RESPONSES resp=$MAX_RESP_LENGTH"
echo "[mopd] train_files=$TRAIN_DATASET"
echo "[mopd] val_files=$TEST_DATASET"
echo "[mopd] student=$ACTOR_MODEL_PATH"
echo "[mopd] math_teacher=$MATH_TEACHER_PATH"
echo "[mopd] code_teacher=$CODE_TEACHER_PATH"

export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_logits_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_n${N_RESPONSES}_b${TRAIN_BATCH_SIZE}_r${MAX_RESP_LENGTH}_$(date +%Y-%m-%d_%H-%M-%S)}
export CKPT_PATH=${PROJECT_PATH}/${EXPERIMENT_NAME}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1

echo "[mopd] proxy cleared; NO_PROXY=$NO_PROXY"
echo "[mopd] ray port=$RAY_PORT temp=$RAY_TEMP_DIR cpus=$RAY_NUM_CPUS object_store=$RAY_OBJECT_STORE_MEMORY"
echo "[mopd] mem: GPU_MEM_UTIL=$GPU_MEM_UTIL RM_USE_DYNAMIC_BSZ=$RM_USE_DYNAMIC_BSZ RM_MICRO_BATCH=$RM_MICRO_BATCH_SIZE_PER_GPU ACTOR_USE_DYNAMIC_BSZ=$ACTOR_USE_DYNAMIC_BSZ ACTOR_MICRO_BATCH=$ACTOR_MICRO_BATCH_SIZE_PER_GPU ACTOR_OFFLOAD=$ACTOR_PARAM_OFFLOAD/$ACTOR_OPTIMIZER_OFFLOAD"

if [ "$SKIP_RAY_STOP" != "1" ]; then
    ray stop --force || true
    pkill -9 -f "ray/.*/gcs_server.*--gcs_server_port=${RAY_PORT}" 2>/dev/null || true
    pkill -9 -f "temp-dir=${RAY_TEMP_DIR}" 2>/dev/null || true
    rm -rf "${RAY_TEMP_DIR}"
fi

RAY_START_EXTRA=()
[ -n "${RAY_OBJECT_STORE_MEMORY:-}" ] && RAY_START_EXTRA+=(--object-store-memory="$RAY_OBJECT_STORE_MEMORY")
[ -n "${RAY_NUM_CPUS:-}" ] && RAY_START_EXTRA+=(--num-cpus="$RAY_NUM_CPUS")
ray start --head --port="$RAY_PORT" --temp-dir="$RAY_TEMP_DIR" --disable-usage-stats "${RAY_START_EXTRA[@]}"
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
sleep 3

PPO_MAX_TOKEN_LEN_PER_GPU=$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768 ))

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.shuffle=True \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.use_rep_distillation=$USE_REP_DISTILLATION \
    +actor_rollout_ref.actor.rep_distillation_only=$REP_DISTILLATION_ONLY \
    +actor_rollout_ref.actor.rep_distillation_coef=${REP_DISTILLATION_COEF:-1.0} \
    +actor_rollout_ref.actor.rep_distillation_positions=${REP_DISTILLATION_POSITIONS:-last_k} \
    +actor_rollout_ref.actor.rep_distillation_last_k=${REP_DISTILLATION_LAST_K:-2000} \
    +actor_rollout_ref.actor.rep_distillation_first_k=${REP_DISTILLATION_FIRST_K:-2000} \
    +actor_rollout_ref.actor.rep_distillation_layers=${REP_DISTILLATION_LAYERS:-all} \
    +actor_rollout_ref.actor.rep_projector_mode=${REP_PROJECTOR_MODE:-full} \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.enable=True \
    reward_model.model.path=$MATH_TEACHER_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.use_dynamic_bsz=$RM_USE_DYNAMIC_BSZ \
    reward_model.micro_batch_size_per_gpu=$RM_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.multi_teacher.enable=True \
    reward_model.multi_teacher.routing_key=ability \
    +reward_model.multi_teacher.teachers.math.path=$MATH_TEACHER_PATH \
    +reward_model.multi_teacher.teachers.code.path=$CODE_TEACHER_PATH \
    custom_reward_function.path="$SCRIPT_DIR/verl/verl/utils/reward_score/mopd_reward.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.log_val_generations=2 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.validation_data_dir=${PROJECT_PATH}/logs/validation_log/$EXPERIMENT_NAME
