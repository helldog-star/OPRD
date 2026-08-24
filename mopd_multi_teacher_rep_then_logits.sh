#!/bin/bash
# Two-stage: naive OPRD (rep-only) then logits OPD.
#
#   Stage1: OPRD layers=all, last_k=2000, 30 steps, new run
#   Stage2: resume actor weights + data.pt + global_steps, NEW optimizer,
#           logits OPD for 20 more steps (WandB x-axis 30 → 50)
#
# Stage2 does not load optimizer / lr_scheduler (checkpoint load_contents=['model']).
# Prompts continue the same shuffled stream as stage1.
#
# Default (the stage-1 run for run_tag 2026-08-23_11-57-02 already exists) is to
# RESUME stage2 (logits OPD) from its global_step_30 checkpoint — actor weights
# + data.pt + global_step, fresh optimizer, x-axis 30 -> 50:
#   bash mopd_multi_teacher_rep_then_logits.sh
#
# Fresh full two-stage run (brand-new stage1): set STAGE=all and a new
# STAGE1_NAME / RUN_TAG so it does not reuse the existing s1 dir:
#   STAGE=all STAGE1_NAME=mopd_rep_then_logits_s1_<newtag> RUN_TAG=<newtag> \
#     bash mopd_multi_teacher_rep_then_logits.sh
#
#   STAGE=1  bash mopd_multi_teacher_rep_then_logits.sh
#   STAGE=2  STAGE1_NAME=... bash ...
#   DRY_RUN=1 bash mopd_multi_teacher_rep_then_logits.sh   # print plan, do not train
#
# Training hparams below are pinned (not ${VAR:-default}) so a leftover
# TEST_FREQ/SAVE_FREQ in the parent shell cannot override this recipe.
# STAGE / DRY_RUN / RUN_TAG / STAGE{1,2}_NAME / WANDB_RUN_GROUP remain overridable.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# export OPRD_CONDA_SH=${OPRD_CONDA_SH:-/root/siton-tmp/home/liuxinyu/miniconda3/etc/profile.d/conda.sh}
# export OPRD_CONDA_ENV=${OPRD_CONDA_ENV:-verl}
# export OPRD_CONDA_BIN=${OPRD_CONDA_BIN:-/root/siton-tmp/home/liuxinyu/miniconda3/envs/verl/bin}
# # shellcheck disable=SC1090
# source "$OPRD_CONDA_SH"
# conda activate "$OPRD_CONDA_ENV"
# export PATH="$OPRD_CONDA_BIN:$PATH"
# export PYTHONPATH="${SCRIPT_DIR}/verl:${PYTHONPATH:-}"

export NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,0.0.0.0,::1,172.17.0.4,172.17.0.0/16}
export no_proxy="$NO_PROXY"
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= ALL_PROXY= all_proxy=

export PROJECT_PATH=${PROJECT_PATH:-./outputs}
export PROJECT_NAME=${PROJECT_NAME:-MOPD_MultiTeacher}
export RAY_PORT=${RAY_PORT:-6399}

STAGE=${STAGE:-2}   # all | 1 | 2  (default 2: resume stage2 from STAGE1_NAME)
DRY_RUN=${DRY_RUN:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}
export STAGE1_NAME=${STAGE1_NAME:-mopd_rep_then_logits_s1_2026-08-23_11-57-02}
export STAGE2_NAME=${STAGE2_NAME:-mopd_rep_then_logits_s2_${RUN_TAG}}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-mopd_rep_then_logits_2026-08-23_11-57-02}

# Pinned recipe (direct assignment; ignores inherited env).
STAGE1_STEPS=30
STAGE2_STEPS=20
STAGE1_CKPT_STEP=$STAGE1_STEPS
STAGE2_END_STEP=$((STAGE1_CKPT_STEP + STAGE2_STEPS))
STAGE1_SAVE_FREQ=10
STAGE2_SAVE_FREQ=10
TEST_FREQ=5
TOTAL_EPOCHS=2
VAL_BEFORE_TRAIN=True
DATA_SEED=42
STAGE1_ACTOR_LR=1e-5
STAGE2_ACTOR_LR=5e-6
STAGE2_GPU_MEM_UTIL=0.55

STAGE1_CKPT_DIR="${PROJECT_PATH}/${STAGE1_NAME}"
RESUME_DIR="${STAGE1_CKPT_DIR}/global_step_${STAGE1_CKPT_STEP}"
STAGE2_CKPT_DIR="${PROJECT_PATH}/${STAGE2_NAME}/global_step_${STAGE2_END_STEP}"
MANIFEST="${PROJECT_PATH}/${STAGE1_NAME}/rep_then_logits_manifest.txt"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    LOG_DIR=${LOG_DIR:-logs}
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/mopd_rep_then_logits_${RUN_TAG}.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "[rep2logits] master log: $LOG_FILE"
fi

_write_manifest() {
    mkdir -p "$(dirname "$MANIFEST")"
    cat > "$MANIFEST" <<EOF
run_tag=$RUN_TAG
wandb_group=$WANDB_RUN_GROUP
stage1_name=$STAGE1_NAME
stage2_name=$STAGE2_NAME
stage1_ckpt_dir=$STAGE1_CKPT_DIR
stage1_ckpt_step=$STAGE1_CKPT_STEP
resume_dir=$RESUME_DIR
stage1_steps=$STAGE1_STEPS
stage2_steps=$STAGE2_STEPS
stage2_end_step=$STAGE2_END_STEP
stage2_actor_lr=$STAGE2_ACTOR_LR
EOF
    echo "[rep2logits] wrote $MANIFEST"
}

# Trainer / Ray / Python 3.12 multiprocess often exit non-zero during
# shutdown after the last checkpoint is already on disk. Continue if the
# expected actor + data.pt are present; fail if they are not.
_run_stage_and_tolerate_shutdown() {
    local stage_name="$1"
    local ckpt_dir="$2"
    shift 2
    local rc=0
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[rep2logits] ${stage_name} trainer exited 0"
        return 0
    fi
    if [ -d "${ckpt_dir}/actor" ] && [ -f "${ckpt_dir}/data.pt" ]; then
        echo "[rep2logits] ${stage_name} exited ${rc} after writing ${ckpt_dir}; treating as shutdown noise"
        return 0
    fi
    echo "[rep2logits] ${stage_name} failed with exit ${rc} and missing actor/data.pt under ${ckpt_dir}" >&2
    ls -la "$ckpt_dir" 2>/dev/null || echo "  (missing $ckpt_dir)" >&2
    exit "$rc"
}

run_stage1() {
    echo "[rep2logits] ===== stage1 OPRD rep-only: steps=$STAGE1_STEPS save=$STAGE1_SAVE_FREQ test=$TEST_FREQ ====="
    echo "[rep2logits] wandb name=$STAGE1_NAME group=$WANDB_RUN_GROUP"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[rep2logits] DRY_RUN skip stage1 launch"
        echo "[rep2logits]   EXPERIMENT_NAME=$STAGE1_NAME TOTAL_TRAINING_STEPS=$STAGE1_STEPS ACTOR_LR=$STAGE1_ACTOR_LR"
        echo "[rep2logits]   USE_REP_DISTILLATION=True REP_DISTILLATION_ONLY=True (via oprd.sh)"
        echo "[rep2logits]   DATA_SEED=$DATA_SEED SAVE_FREQ=$STAGE1_SAVE_FREQ TEST_FREQ=$TEST_FREQ"
        return 0
    fi
    _run_stage_and_tolerate_shutdown "stage1" "$RESUME_DIR" \
        env \
            EXPERIMENT_NAME="$STAGE1_NAME" \
            MOPD_LOG_PREFIX="mopd_rep_then_logits_s1" \
            WANDB_RUN_GROUP="$WANDB_RUN_GROUP" \
            WANDB_TAGS="rep_then_logits,stage1,oprd" \
            TOTAL_TRAINING_STEPS="$STAGE1_STEPS" \
            TOTAL_EPOCHS="$TOTAL_EPOCHS" \
            SAVE_FREQ="$STAGE1_SAVE_FREQ" \
            TEST_FREQ="$TEST_FREQ" \
            VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
            ACTOR_LR="$STAGE1_ACTOR_LR" \
            DATA_SEED="$DATA_SEED" \
            bash "$SCRIPT_DIR/mopd_multi_teacher_oprd.sh"
    echo "[rep2logits] stage1 done. ckpt dir=$STAGE1_CKPT_DIR"
}

run_stage2() {
    if [ "$DRY_RUN" != "1" ]; then
        if [ ! -d "${RESUME_DIR}/actor" ] || [ ! -f "${RESUME_DIR}/data.pt" ]; then
            echo "[rep2logits] stage2 needs actor + data.pt under $RESUME_DIR" >&2
            ls -la "$RESUME_DIR" 2>/dev/null || echo "  (missing $RESUME_DIR)" >&2
            exit 1
        fi
    elif [ ! -d "${RESUME_DIR}/actor" ] || [ ! -f "${RESUME_DIR}/data.pt" ]; then
        echo "[rep2logits] DRY_RUN: resume dir not present yet (expected before stage1): $RESUME_DIR"
    fi
    local resume_abs="$RESUME_DIR"
    if [ -d "$RESUME_DIR" ]; then
        resume_abs="$(cd "$RESUME_DIR" && pwd)"
    fi
    echo "[rep2logits] ===== stage2 logits OPD: steps ${STAGE1_CKPT_STEP} → ${STAGE2_END_STEP}, lr=$STAGE2_ACTOR_LR ====="
    echo "[rep2logits] resume=$resume_abs (model + data.pt + step; no optimizer)"
    echo "[rep2logits] wandb name=$STAGE2_NAME group=$WANDB_RUN_GROUP"
    echo "[rep2logits]   RESUME_FROM_PATH=$resume_abs"
    echo "[rep2logits]   CKPT_LOAD_CONTENTS=['model']"
    echo "[rep2logits]   TOTAL_TRAINING_STEPS=$STAGE2_END_STEP ACTOR_LR=$STAGE2_ACTOR_LR"
    echo "[rep2logits]   USE_REP_DISTILLATION=False REP_DISTILLATION_ONLY=False"
    if [ "$DRY_RUN" = "1" ]; then
        echo "[rep2logits] DRY_RUN skip stage2 launch"
        return 0
    fi
    _run_stage_and_tolerate_shutdown "stage2" "$STAGE2_CKPT_DIR" \
        env -u USE_REP_DISTILLATION -u REP_DISTILLATION_ONLY -u EXPERIMENT_NAME -u MOPD_LOG_PREFIX \
            -u TOTAL_TRAINING_STEPS -u SAVE_FREQ -u ACTOR_LR -u GPU_MEM_UTIL \
            -u RESUME_FROM_PATH -u CKPT_LOAD_CONTENTS \
            EXPERIMENT_NAME="$STAGE2_NAME" \
            MOPD_LOG_PREFIX="mopd_rep_then_logits_s2" \
            WANDB_RUN_GROUP="$WANDB_RUN_GROUP" \
            WANDB_TAGS="rep_then_logits,stage2,logits" \
            USE_REP_DISTILLATION=False \
            REP_DISTILLATION_ONLY=False \
            RESUME_FROM_PATH="$resume_abs" \
            CKPT_LOAD_CONTENTS="['model']" \
            TOTAL_TRAINING_STEPS="$STAGE2_END_STEP" \
            TOTAL_EPOCHS="$TOTAL_EPOCHS" \
            SAVE_FREQ="$STAGE2_SAVE_FREQ" \
            TEST_FREQ="$TEST_FREQ" \
            VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
            ACTOR_LR="$STAGE2_ACTOR_LR" \
            DATA_SEED="$DATA_SEED" \
            GPU_MEM_UTIL="$STAGE2_GPU_MEM_UTIL" \
            bash "$SCRIPT_DIR/mopd_multi_teacher_logits.sh"
    echo "[rep2logits] stage2 done. ckpt dir=${PROJECT_PATH}/${STAGE2_NAME}"
}

echo "[rep2logits] STAGE=$STAGE DRY_RUN=$DRY_RUN run_tag=$RUN_TAG wandb_group=$WANDB_RUN_GROUP"
echo "[rep2logits] s1=$STAGE1_NAME (1–${STAGE1_STEPS}) → resume step $STAGE1_CKPT_STEP → s2=$STAGE2_NAME (${STAGE1_CKPT_STEP}–${STAGE2_END_STEP}, lr=$STAGE2_ACTOR_LR)"
_write_manifest

case "$STAGE" in
    all)
        run_stage1
        run_stage2
        ;;
    1|s1|stage1)
        run_stage1
        ;;
    2|s2|stage2)
        run_stage2
        ;;
    *)
        echo "[rep2logits] unknown STAGE=$STAGE (use all|1|2)" >&2
        exit 1
        ;;
esac

echo "[rep2logits] finished STAGE=$STAGE"
echo "[rep2logits] manifest=$MANIFEST"

# nohup bash mopd_multi_teacher_rep_then_logits.sh > mopd_multi_teacher_rep_then_logits.log 2>&1 &
