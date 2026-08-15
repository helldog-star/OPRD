#!/bin/bash
# Download G-OPD training data + Step1200 teachers for MOPD multi-teacher runs.
#
#   bash scripts/download_mopd_assets.sh
set -eo pipefail

MODEL_DIR=${MODEL_DIR:-/root/siton-tmp/home/liuxinyu/hf_models}
DATA_DIR=${DATA_DIR:-/root/siton-tmp/home/liuxinyu/hf_datasets}
GOPD_DATA_DIR=${GOPD_DATA_DIR:-$DATA_DIR/G-OPD-Training-Data}

mkdir -p "$MODEL_DIR" "$GOPD_DATA_DIR"

echo "[download] G-OPD-Training-Data -> $GOPD_DATA_DIR"
huggingface-cli download Keven16/G-OPD-Training-Data \
  --repo-type dataset \
  --local-dir "$GOPD_DATA_DIR"

echo "[download] Math teacher Step1200"
huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 \
  --local-dir "$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step1200"

echo "[download] Code teacher Step1200"
huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 \
  --local-dir "$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Code-Step1200"

echo "[download] Student Qwen3-4B (skip if present)"
if [ ! -f "$MODEL_DIR/Qwen3-4B/config.json" ]; then
  huggingface-cli download Qwen/Qwen3-4B --local-dir "$MODEL_DIR/Qwen3-4B"
fi

echo "[download] done."
echo "  math parquet: $GOPD_DATA_DIR/DeepMath-103K/train_filtered_level6.parquet"
echo "  code parquet: $GOPD_DATA_DIR/Eurus/code_train.parquet"
echo "  math teacher: $MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step1200"
echo "  code teacher: $MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Code-Step1200"
