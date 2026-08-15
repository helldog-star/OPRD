# MOPD 多专家融合实验说明

本目录脚本用于在 **Qwen3-4B** 上做双 teacher 的多专家 on-policy 蒸馏对比：

| 对照实验 | 启动脚本 | Teacher 信号 |
|----|----------|-------------|
| **MOPD-logits** | `mopd_multi_teacher_logits.sh` | token reverse-KL（`token_reward_direct`，`LOG_PROB_TOP_K=0`） |
| **MOPD-OPRD** | `mopd_multi_teacher_oprd.sh` | 表征 MSE（rep-only，`last_k=2000`，`layers=all`） |

两边共享：**同一 student、同一对 Step1200 teacher、同一混合数据、同一 `ability` 路由、同一 batch / n / max length**。

参考：

- [MOPD: Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2606.30406v1)（多 teacher 路由 + on-policy 蒸馏；正式设定 `N=1`、大 batch）
- [G-OPD](https://arxiv.org/abs/2602.12125)（同数据/同规模 4B 蒸馏超参；Table 6：`batch=1024`，`n=1`，`resp=16384`）
- 训练数据：[Keven16/G-OPD-Training-Data](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data)
  - Math：[DeepMath-103K/train_filtered_level6.parquet](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data/tree/main/DeepMath-103K)
  - Code：[Eurus/code_train.parquet](https://huggingface.co/datasets/Keven16/G-OPD-Training-Data/tree/main/Eurus)

---

## 0. 资源与默认超参（8×A100）

| 项 | 默认值 | 说明 |
|----|--------|------|
| GPUs | 8 | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| Student | `Qwen3-4B` | |
| Math teacher | `Qwen3-4B-Non-Thinking-RL-Math-Step1200` | HF: `Keven16/...-Math-Step1200` |
| Code teacher | `Qwen3-4B-Non-Thinking-RL-Code-Step1200` | HF: `Keven16/...-Code-Step1200` |
| `MAX_PROMPT_LENGTH` | 2048 | 与 G-OPD 一致 |
| `MAX_RESP_LENGTH` | **16384** | 对齐 math teacher / G-OPD distill；code RL 原为 8192，蒸馏统一用 16384 |
| `TRAIN_BATCH_SIZE` | **1024** | prompts / step（G-OPD Table 6） |
| `MINI_BATCH_SIZE` | 128 | PPO mini-batch |
| `N_RESPONSES` | **1** | MOPD / G-OPD distill 均用 `n=1` |
| `ACTOR_LR` | `1e-5` | |
| `TOTAL_TRAINING_STEPS` | 200 | 可按机器时间改为 50–500；G-OPD same-size 曾用 50 |
| Thinking | `enable_thinking=False` | Non-Thinking 模型必须关闭 |

Teacher 自身 RL 训练长度（G-OPD 附录，供对照）：Math GRPO **16384** / Code GRPO **8192**；步数 Step1200 比 Step500/300 训得更久。

单卡冒烟请用独立脚本（见 §4.3），不要改主脚本默认。

---

## 1. 环境

```bash
# 按机器修改路径
export MODEL_DIR=/path/to/hf_models
export DATA_DIR=/path/to/hf_datasets
export GOPD_DATA_DIR=$DATA_DIR/G-OPD-Training-Data

# conda / PYTHONPATH（脚本内也有默认 siton 路径，可 export 覆盖）
export OPRD_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
export OPRD_CONDA_ENV=verl
export OPRD_CONDA_BIN=/path/to/miniconda3/envs/verl/bin

cd /path/to/OPRD
```

Ray 启动前建议清掉代理（尤其是 `ALL_PROXY=socks5://...`），否则 GCS 可能卡住：

```bash
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1
```

---

## 2. 下载模型与数据

```bash
bash scripts/download_mopd_assets.sh
```

会下载到：

```text
$DATA_DIR/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet
$DATA_DIR/G-OPD-Training-Data/Eurus/code_train.parquet
$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step1200/
$MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Code-Step1200/
$MODEL_DIR/Qwen3-4B/          # 若尚不存在
```

也可手动：

```bash
huggingface-cli download Keven16/G-OPD-Training-Data \
  --repo-type dataset --local-dir $GOPD_DATA_DIR

huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 \
  --local-dir $MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step1200

huggingface-cli download Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 \
  --local-dir $MODEL_DIR/Qwen3-4B-Non-Thinking-RL-Code-Step1200
```

---

## 3. 数据预处理（混合 train parquet）

路由键是行级字段 **`ability ∈ {math, code}`**。预处理脚本只保留 verl 需要的列并打标：

```bash
# 默认：math:code = 0.5:0.5（按下采样较多一侧，受较少一侧数量限制）
python scripts/prepare_mopd_mix.py \
  --math $GOPD_DATA_DIR/DeepMath-103K/train_filtered_level6.parquet \
  --code $GOPD_DATA_DIR/Eurus/code_train.parquet \
  --out datasets/mopd_math_code_mix_balanced.parquet

# 可选：不做平衡，全量拼接（~7:3 自然比例）
python scripts/prepare_mopd_mix.py --no-balance \
  --out datasets/mopd_math_code_mix_full.parquet

# 可选：小规模冒烟
python scripts/prepare_mopd_mix.py --n_math 4000 --n_code 4000 \
  --out datasets/mopd_math_code_mix_8k.parquet
```

验证集使用仓库内预构建的 `datasets/test_data/mopd_val_mix.parquet`（AIME24 + MATH-500 子集 + code 子集）。如需重做：

```bash
python scripts/prepare_mopd_val_mix.py   # 若存在；或沿用现有 parquet
```

训练脚本在找不到 `MIX_PARQUET` 时会自动调用 `prepare_mopd_mix.py`。

---

## 4. 启动训练（8×A100）

### 4.1 MOPD-logits

```bash
unset ALL_PROXY all_proxy HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_PORT=6399
export TOTAL_TRAINING_STEPS=200
export TEST_FREQ=20
export SAVE_FREQ=50
export EXPERIMENT_NAME=mopd_logits_$(date +%Y-%m-%d_%H-%M-%S)

# 可选覆盖
# export TRAIN_BATCH_SIZE=1024 N_RESPONSES=1 MAX_RESP_LENGTH=16384
# export MIX_PARQUET=$PWD/datasets/mopd_math_code_mix_full.parquet  # 若要用未平衡全量

nohup bash mopd_multi_teacher_logits.sh >/dev/null 2>&1 &
# 日志：logs/mopd_logits_*.log
```

### 4.2 MOPD-OPRD（表征蒸馏对照）

```bash
export RAY_PORT=6400          # 与 logits 并行时换端口
export EXPERIMENT_NAME=mopd_oprd_$(date +%Y-%m-%d_%H-%M-%S)
# 与 logits 对齐的关键：不要单独改 MAX_RESP_LENGTH / TRAIN_BATCH_SIZE / N_RESPONSES

nohup bash mopd_multi_teacher_oprd.sh >/dev/null 2>&1 &
# 日志：logs/mopd_oprd_*.log
```

若 OPRD 仍 OOM，可先加系统向缓解（不改长度）：

```bash
export ACTOR_PARAM_OFFLOAD=True ACTOR_OPTIMIZER_OFFLOAD=True
export GPU_MEM_UTIL=0.40
export RM_USE_DYNAMIC_BSZ=False ACTOR_USE_DYNAMIC_BSZ=False
# 或略降表征宽度：export REP_DISTILLATION_LAST_K=1024
```

### 4.3 单卡冒烟测试

```bash
# logits
bash mopd_multi_teacher_logits_1gpu.sh

# oprd（含 1 卡 OOM 缓解：offload + 关 dynamic bsz）
bash mopd_multi_teacher_oprd_1gpu.sh
```

单卡默认：`resp=4096`，`batch=8`，`n=4`，小混合集 `mopd_math_code_mix_8k.parquet`。**不能与 8 卡正式结果直接对比。**

---