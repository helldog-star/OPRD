import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# 路径
# ============================================================

ROOT = Path("/mnt/yetenglong/OPRD")
sys.path.insert(0, str(ROOT / "verl"))

from verl.utils.reward_score.mopd_reward import reward_func


MODELS = {
    "Qwen3-4B":
        "/mnt/yetenglong/hf_models/Qwen3-4B",

    "Math-Step1200":
        "/mnt/yetenglong/hf_models/"
        "Qwen3-4B-Non-Thinking-RL-Math-Step1200",

    "Code-Step1200":
        "/mnt/yetenglong/hf_models/"
        "Qwen3-4B-Non-Thinking-RL-Code-Step1200",
}


DATASETS = {
    "AIME24":
        "/mnt/yetenglong/OPRD/"
        "datasets/test_data/AIME24/test.parquet",

    "MATH-500":
        "/mnt/yetenglong/OPRD/"
        "datasets/test_data/MATH-500/test.parquet",

    "Eurus-Code":
        "/mnt/yetenglong/datasets/"
        "Eurus/Eurus/code_validation.parquet",
}


MATH_SUFFIX = (
    " Please reason step by step, and put your final answer "
    "within \\boxed{}."
)


# ============================================================
# 参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS.keys()),
        choices=list(DATASETS.keys()),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=32,
        help="每道题生成多少个 response",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0=完整数据集",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="每批多少道题；n=32 时建议 1",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "eval_results"),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除当前配置已有结果并重新开始",
    )

    parser.add_argument(
        "--child-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


# ============================================================
# 数据
# ============================================================

def normalize_messages(prompt):
    if isinstance(prompt, str):
        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    messages = []

    for msg in prompt:
        messages.append(
            {
                "role": str(msg.get("role", "user")),
                "content": str(msg.get("content", "")),
            }
        )

    return messages


def load_dataset(name, path, limit=0):
    df = pd.read_parquet(path)

    if limit > 0:
        df = df.iloc[:limit]

    samples = []

    for position, (_, row) in enumerate(df.iterrows()):

        messages = normalize_messages(row["prompt"])

        data_source = str(row["data_source"])
        ability = str(row["ability"])

        reward_model = row["reward_model"]

        if isinstance(reward_model, dict):
            ground_truth = reward_model.get(
                "ground_truth",
                "",
            )
        else:
            try:
                ground_truth = reward_model["ground_truth"]
            except Exception:
                ground_truth = ""

        extra_info = row["extra_info"]

        if isinstance(extra_info, dict):
            extra_info = dict(extra_info)
        else:
            try:
                extra_info = dict(extra_info)
            except Exception:
                extra_info = {}

        # 帮助 mopd_reward 识别 math / code
        extra_info["ability"] = ability

        if "id" in row.index:
            raw_id = row["id"]

            try:
                missing = pd.isna(raw_id)
            except Exception:
                missing = False

            if missing:
                raw_id = None
        else:
            raw_id = None

        if raw_id is None:
            raw_id = extra_info.get(
                "index",
                position,
            )

        samples.append(
            {
                "example_index": position,
                "id": str(raw_id),
                "messages": messages,
                "data_source": data_source,
                "ability": ability,
                "ground_truth": ground_truth,
                "extra_info": extra_info,
            }
        )

    print(
        f"[DATA] {name}: "
        f"{len(samples)} examples"
    )

    return samples


# ============================================================
# Prompt
# ============================================================

def build_prompt(sample, tokenizer):
    messages = [
        dict(x)
        for x in sample["messages"]
    ]

    if sample["ability"] == "math":
        messages[-1]["content"] = (
            messages[-1]["content"].rstrip()
            + MATH_SUFFIX
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================
# Reward
# ============================================================

def score_response(sample, response):
    try:
        result = reward_func(
            data_source=sample["data_source"],
            solution_str=response,
            ground_truth=sample["ground_truth"],
            extra_info=sample["extra_info"],
        )

        if isinstance(result, dict):
            score = float(
                result.get("score", 0.0)
            )

            acc = float(
                result.get(
                    "acc",
                    score,
                )
            )

            format_score = float(
                result.get(
                    "format_score",
                    0.0,
                )
            )

            return {
                "score": score,
                "acc": acc,
                "format_score": format_score,
                "pred": result.get("pred", ""),
                "extracted_gt": result.get(
                    "extracted_gt",
                    "",
                ),
                "reward_error": None,
            }

        value = float(result)

        return {
            "score": value,
            "acc": value,
            "format_score": 0.0,
            "pred": "",
            "extracted_gt": str(
                sample["ground_truth"]
            ),
            "reward_error": None,
        }

    except Exception as e:
        print(
            "[REWARD ERROR]",
            sample["id"],
            repr(e),
        )

        return {
            "score": 0.0,
            "acc": 0.0,
            "format_score": 0.0,
            "pred": "",
            "extracted_gt": str(
                sample["ground_truth"]
            ),
            "reward_error": repr(e),
        }


# ============================================================
# 运行配置 / Resume
# ============================================================

def make_run_config(
    model_name,
    dataset_name,
    args,
):
    return {
        "model": model_name,
        "model_path": MODELS[model_name],
        "dataset": dataset_name,
        "dataset_path": DATASETS[dataset_name],
        "n": args.n,
        "limit": args.limit,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "enable_thinking": False,
    }


def prepare_output(
    out_file,
    config_file,
    summary_file,
    config,
    n,
    overwrite,
):
    if overwrite:
        for path in [
            out_file,
            config_file,
            summary_file,
        ]:
            if path.exists():
                path.unlink()

    if config_file.exists():
        old_config = json.loads(
            config_file.read_text(
                encoding="utf-8"
            )
        )

        if old_config != config:
            raise RuntimeError(
                "\n已有结果的配置和当前配置不同。\n"
                "如果确认要重新跑，请加 --overwrite\n"
            )

    elif out_file.exists():
        raise RuntimeError(
            "\n发现旧 jsonl，但没有 run_config。\n"
            "这是之前 n=1 测试留下的结果。\n"
            "正式评测请加 --overwrite。\n"
        )

    config_file.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not out_file.exists():
        return set()

    # --------------------------------------------------------
    # 检查已经完整完成的 example
    # --------------------------------------------------------

    counts = Counter()

    with out_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            try:
                record = json.loads(line)
            except Exception:
                continue

            idx = str(
                record["example_index"]
            )

            counts[idx] += 1

    completed = {
        idx
        for idx, count in counts.items()
        if count >= n
    }

    # --------------------------------------------------------
    # 如果某一道题只写了一部分 response，
    # 删除这一题的残缺记录，后面重新生成。
    # 同时防止重复超过 n。
    # --------------------------------------------------------

    if any(count != n for count in counts.values()):
        tmp_file = out_file.with_suffix(
            ".jsonl.tmp"
        )

        kept = Counter()

        with out_file.open(
            "r",
            encoding="utf-8",
        ) as src, tmp_file.open(
            "w",
            encoding="utf-8",
        ) as dst:

            for line in src:
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                idx = str(
                    record["example_index"]
                )

                if idx not in completed:
                    continue

                if kept[idx] >= n:
                    continue

                dst.write(line)
                kept[idx] += 1

        tmp_file.replace(out_file)

    print(
        f"[RESUME] completed examples: "
        f"{len(completed)}"
    )

    return completed


# ============================================================
# Summary
# ============================================================

def summarize_jsonl(
    out_file,
    model_name,
    dataset_name,
    args,
):
    per_example = defaultdict(
        lambda: {
            "count": 0,
            "reward_sum": 0.0,
            "acc_sum": 0.0,
            "format_sum": 0.0,
            "first_reward": None,
            "first_exact": None,
            "best_reward": 0.0,
            "exact_count": 0,
            "reward_errors": 0,
        }
    )

    total_responses = 0
    total_reward = 0.0
    total_acc = 0.0
    total_format = 0.0
    total_exact = 0
    total_errors = 0

    with out_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            record = json.loads(line)

            idx = str(
                record["example_index"]
            )

            response_id = int(
                record["response_id"]
            )

            score = float(
                record["score"]
            )

            acc = float(
                record["acc"]
            )

            format_score = float(
                record["format_score"]
            )

            exact = (
                1
                if score >= 0.999999
                else 0
            )

            error = (
                record.get("reward_error")
                is not None
            )

            stat = per_example[idx]

            stat["count"] += 1
            stat["reward_sum"] += score
            stat["acc_sum"] += acc
            stat["format_sum"] += format_score

            stat["best_reward"] = max(
                stat["best_reward"],
                score,
            )

            stat["exact_count"] += exact

            if error:
                stat["reward_errors"] += 1

            if response_id == 0:
                stat["first_reward"] = score
                stat["first_exact"] = exact

            total_responses += 1
            total_reward += score
            total_acc += acc
            total_format += format_score
            total_exact += exact

            if error:
                total_errors += 1

    num_examples = len(per_example)

    if num_examples == 0:
        return {}

    example_mean_rewards = []
    best_rewards = []
    pass_at_n = []
    first_rewards = []
    first_exact = []

    for stat in per_example.values():

        if stat["count"] > 0:
            example_mean_rewards.append(
                stat["reward_sum"]
                / stat["count"]
            )

        best_rewards.append(
            stat["best_reward"]
        )

        pass_at_n.append(
            1.0
            if stat["exact_count"] > 0
            else 0.0
        )

        if stat["first_reward"] is not None:
            first_rewards.append(
                stat["first_reward"]
            )

        if stat["first_exact"] is not None:
            first_exact.append(
                float(stat["first_exact"])
            )

    def mean(values):
        if not values:
            return 0.0

        return sum(values) / len(values)

    summary = {
        "model": model_name,
        "dataset": dataset_name,

        "num_examples": num_examples,
        "n": args.n,

        "num_responses":
            total_responses,

        # 最主要：所有 response 的平均 reward
        "mean_reward":
            total_reward / total_responses,

        "mean_acc":
            total_acc / total_responses,

        "mean_format_score":
            total_format / total_responses,

        # 每一道题32次 reward 先平均，再跨题平均
        "mean_example_reward":
            mean(example_mean_rewards),

        # 所有 response 中完全正确的比例
        "exact_match_rate":
            total_exact / total_responses,

        # 每题第0个 response 的表现
        "first_response_mean_reward":
            mean(first_rewards),

        "pass_at_1":
            mean(first_exact),

        # 每题32次里面最佳 reward
        "best_of_n_mean_reward":
            mean(best_rewards),

        # 32次里只要有一次完全正确
        "pass_at_n":
            mean(pass_at_n),

        "reward_errors":
            total_errors,

        "temperature":
            args.temperature,

        "top_p":
            args.top_p,

        "max_tokens":
            args.max_tokens,

        "max_model_len":
            args.max_model_len,

        "seed":
            args.seed,

        "enable_thinking":
            False,
    }

    return summary


# ============================================================
# 一个数据集
# ============================================================

def evaluate_dataset(
    llm,
    tokenizer,
    model_name,
    dataset_name,
    args,
):
    from vllm import SamplingParams

    output_root = Path(
        args.output_dir
    )

    model_dir = (
        output_root / model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_file = (
        model_dir
        / f"{dataset_name}.jsonl"
    )

    config_file = (
        model_dir
        / f"{dataset_name}_run_config.json"
    )

    summary_file = (
        model_dir
        / f"{dataset_name}_summary.json"
    )

    config = make_run_config(
        model_name,
        dataset_name,
        args,
    )

    completed = prepare_output(
        out_file=out_file,
        config_file=config_file,
        summary_file=summary_file,
        config=config,
        n=args.n,
        overwrite=args.overwrite,
    )

    samples = load_dataset(
        dataset_name,
        DATASETS[dataset_name],
        args.limit,
    )

    remaining = [
        sample
        for sample in samples
        if str(
            sample["example_index"]
        ) not in completed
    ]

    print(
        f"[TODO] {dataset_name}: "
        f"{len(remaining)} examples"
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # --------------------------------------------------------
    # 推理
    # --------------------------------------------------------

    for start in range(
        0,
        len(remaining),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(remaining),
        )

        batch = remaining[start:end]

        print(
            "\n"
            f"[{model_name}] "
            f"[{dataset_name}] "
            f"{start}:{end}/"
            f"{len(remaining)}"
        )

        prompts = [
            build_prompt(
                sample,
                tokenizer,
            )
            for sample in batch
        ]

        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
        )

        batch_rewards = []

        with out_file.open(
            "a",
            encoding="utf-8",
        ) as f:

            for sample, output in zip(
                batch,
                outputs,
            ):

                if len(output.outputs) != args.n:
                    raise RuntimeError(
                        f"Expected {args.n} outputs, "
                        f"got {len(output.outputs)}"
                    )

                # ============================================
                # 每一道题的32个 response 全部打分
                # ============================================

                for response_id, candidate in enumerate(
                    output.outputs
                ):
                    response = candidate.text

                    reward = score_response(
                        sample,
                        response,
                    )

                    batch_rewards.append(
                        reward["score"]
                    )

                    record = {
                        "model":
                            model_name,

                        "dataset":
                            dataset_name,

                        "example_index":
                            sample[
                                "example_index"
                            ],

                        "id":
                            sample["id"],

                        "response_id":
                            response_id,

                        "data_source":
                            sample[
                                "data_source"
                            ],

                        "ability":
                            sample["ability"],

                        "ground_truth":
                            sample[
                                "ground_truth"
                            ],

                        "response":
                            response,

                        "finish_reason":
                            getattr(
                                candidate,
                                "finish_reason",
                                None,
                            ),

                        "num_output_tokens":
                            len(
                                getattr(
                                    candidate,
                                    "token_ids",
                                    [],
                                )
                            ),

                        **reward,
                    }

                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )

                # 一道题32个 response 写完立即 flush
                f.flush()

        if batch_rewards:
            print(
                "[BATCH] mean_reward="
                f"{sum(batch_rewards)/len(batch_rewards):.4f}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_jsonl(
        out_file,
        model_name,
        dataset_name,
        args,
    )

    summary_file.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 80)

    return summary


# ============================================================
# 单个模型
# ============================================================

def evaluate_one_model(
    model_name,
    args,
):
    import torch
    from vllm import LLM

    model_path = MODELS[
        model_name
    ]

    if not Path(
        model_path
    ).exists():
        raise FileNotFoundError(
            model_path
        )

    print("\n" + "#" * 80)
    print("MODEL:", model_name)
    print("PATH :", model_path)
    print("#" * 80)

    llm = LLM(
        model=model_path,
        tokenizer=model_path,

        trust_remote_code=True,

        tensor_parallel_size=1,

        dtype="bfloat16",

        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),

        max_model_len=(
            args.max_model_len
        ),

        max_num_seqs=(
            args.max_num_seqs
        ),

        seed=args.seed,
    )

    tokenizer = llm.get_tokenizer()

    for dataset_name in args.datasets:
        evaluate_dataset(
            llm=llm,
            tokenizer=tokenizer,
            model_name=model_name,
            dataset_name=dataset_name,
            args=args,
        )

    del tokenizer
    del llm

    torch.cuda.empty_cache()


# ============================================================
# 汇总三个模型
# ============================================================

def rebuild_global_summary(
    output_dir
):
    output_root = Path(
        output_dir
    )

    summaries = []

    for summary_file in output_root.glob(
        "*/*_summary.json"
    ):
        try:
            summary = json.loads(
                summary_file.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        if summary:
            summaries.append(
                summary
            )

    if not summaries:
        return

    summaries = sorted(
        summaries,
        key=lambda x: (
            x.get("model", ""),
            x.get("dataset", ""),
        ),
    )

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summaries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        summaries
    ).to_csv(
        output_root
        / "summary.csv",
        index=False,
    )

    print(
        "\nGLOBAL SUMMARY:",
        output_root / "summary.csv"
    )


# ============================================================
# 多模型用独立进程运行
# ============================================================

def run_models_in_subprocesses(
    args
):
    script = str(
        Path(__file__).resolve()
    )

    for model_name in args.models:

        cmd = [
            sys.executable,
            script,

            "--models",
            model_name,

            "--datasets",
            *args.datasets,

            "--n",
            str(args.n),

            "--limit",
            str(args.limit),

            "--batch-size",
            str(args.batch_size),

            "--temperature",
            str(args.temperature),

            "--top-p",
            str(args.top_p),

            "--max-tokens",
            str(args.max_tokens),

            "--max-model-len",
            str(args.max_model_len),

            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),

            "--max-num-seqs",
            str(args.max_num_seqs),

            "--seed",
            str(args.seed),

            "--output-dir",
            args.output_dir,

            "--child-process",
        ]

        if args.overwrite:
            cmd.append(
                "--overwrite"
            )

        print(
            "\nLaunching:",
            " ".join(cmd),
        )

        subprocess.run(
            cmd,
            check=True,
        )

    rebuild_global_summary(
        args.output_dir
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    Path(
        args.output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # 三个模型时，每个模型一个新的 Python/vLLM 进程
    # 避免换模型显存清理不彻底
    if (
        len(args.models) > 1
        and not args.child_process
    ):
        run_models_in_subprocesses(
            args
        )
        return

    if len(args.models) != 1:
        raise RuntimeError(
            "child process must receive exactly one model"
        )

    evaluate_one_model(
        args.models[0],
        args,
    )

    rebuild_global_summary(
        args.output_dir
    )

    print("\nALL DONE")
    print(
        "Results:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()