
# OPRD Baseline Evaluation 实验说明

本目录脚本用于在 **Qwen3-4B 系列模型** 上进行 baseline evaluation。

## Models

  模型                                     类型
  ---------------------------------------- ---------------
  Qwen3-4B                                 Base Model
  Qwen3-4B-Non-Thinking-RL-Math-Step1200   Math 专家模型
  Qwen3-4B-Non-Thinking-RL-Code-Step1200   Code 专家模型

## Evaluation Pipeline

流程：

Model → vLLM generation → OPRD reward function → Score calculation

Reward 使用原 OPRD 项目：

    verl/utils/reward_score/mopd_reward.py

## Benchmarks

评测数据集：

-   AIME24
-   MATH-500
-   Eurus-Code

Eurus-Code 分为：

-   Codeforces
-   CodeContests
-   TACO
-   APPS

## Files

### eval_vllm_reward.py

功能：

-   加载模型
-   vLLM 推理生成 response
-   调用 reward function
-   保存 JSONL 结果
需要：
    输出名称符合calculate_score.py的要求

### calculate_score.py

功能：

-   AIME24 / MATH-500 输出 score 和 accuracy
-   Eurus-Code 输出四个子集及 overall 分数

## Environment

需要：

-   Python \>= 3.10
-   PyTorch
-   vLLM
-   transformers
-   verl

并保证：

    verl/utils/reward_score/mopd_reward.py

存在。

## Path Configuration

运行前修改：

    eval_vllm_reward.py

中的：

``` python
MODELS = {
    "Qwen3-4B": "/path/to/model",
    "Math-Step1200": "/path/to/model",
    "Code-Step1200": "/path/to/model",
}

DATASETS = {
    "AIME24": "/path/to/AIME24/test.parquet",
    "MATH-500": "/path/to/MATH-500/test.parquet",
    "Eurus-Code": "/path/to/Eurus/code_validation.parquet",
}
```

## Running

单模型：

``` bash
python eval_vllm_reward.py     --models Qwen3-4B     --datasets AIME24 MATH-500 Eurus-Code
```

多模型：

``` bash
python eval_vllm_reward.py     --models Qwen3-4B Math-Step1200 Code-Step1200     --datasets AIME24 MATH-500 Eurus-Code
```

计算分数：

``` bash
python calculate_score.py
```

## Notes

-   仅包含 baseline evaluation code。
-   不包含模型权重。
-   不包含数据集。
-   不包含生成结果。
-   reward 实现沿用 OPRD 原项目。
-   分片推理和结果合并不包含在 baseline 代码中。
