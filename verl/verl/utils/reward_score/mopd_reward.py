# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# MOPD math+code reward: route by data_source (and optional ability in extra_info).
# - math (DeepMath / AIME / MATH-500 / ...): ttrl_math boxed grading
# - code (codecontests / apps / taco / codeforces): prime_code (or sandbox_fusion)

from __future__ import annotations

import json
import traceback

from verl.utils.reward_score.ttrl_math import compute_score as math_compute_score

CODE_DATA_SOURCES = {"codecontests", "apps", "codeforces", "taco"}


def _is_code_sample(data_source: str, ground_truth, extra_info=None) -> bool:
    ds = (data_source or "").strip()
    if ds in CODE_DATA_SOURCES:
        return True
    ability = None
    if isinstance(extra_info, dict):
        ability = extra_info.get("ability")
    if ability == "code":
        return True
    if ability == "math":
        return False
    # Fallback: Eurus-style JSON test cases.
    gt = ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except Exception:
            return False
    return isinstance(gt, dict) and "inputs" in gt and "outputs" in gt


def _code_score(
    solution_str: str,
    ground_truth,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
) -> dict:
    has_fence = "```python" in solution_str or "```" in solution_str
    format_score = 1.0 if has_fence else 0.0
    # Keep keys identical to ttrl_math so val aggregation lengths match.
    pred = ""
    if "```python" in solution_str:
        pred = solution_str.split("```python")[-1].split("```")[0].strip()
    elif "```" in solution_str:
        pred = solution_str.split("```")[1].strip() if solution_str.count("```") >= 2 else ""

    try:
        if sandbox_fusion_url:
            from verl.utils.reward_score import sandbox_fusion

            res = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                memory_limit_mb,
                solution_str,
                ground_truth,
                continuous=True,
            )
            if isinstance(res, dict):
                score = float(res.get("score", 0.0))
            elif isinstance(res, (tuple, list)):
                score = float(res[0])
            else:
                score = float(res)
        else:
            from verl.utils.reward_score import prime_code

            success, _meta = prime_code.compute_score(solution_str, ground_truth, continuous=True)
            score = float(success)
    except Exception as e:
        print(f"[mopd_reward] code scoring failed: {e}")
        traceback.print_exc()
        score = 0.0

    return {
        "score": score,
        "acc": score,
        "format_score": format_score,
        "extracted_gt": ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth),
        "pred": pred,
    }


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Sample-level reward for MOPD math+code mixes."""
    try:
        if _is_code_sample(data_source, ground_truth, extra_info):
            return _code_score(
                solution_str,
                ground_truth,
                sandbox_fusion_url=sandbox_fusion_url,
                concurrent_semaphore=concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )

        res = math_compute_score(solution_str, str(ground_truth))
        if isinstance(res, dict):
            # Ensure required keys always exist for mixed math/code validation.
            res.setdefault("score", 0.0)
            res.setdefault("acc", res["score"])
            res.setdefault("format_score", 0.0)
            res.setdefault("extracted_gt", str(ground_truth))
            res.setdefault("pred", "")
            return res
        return {
            "score": float(res),
            "acc": float(res),
            "format_score": 1.0,
            "extracted_gt": str(ground_truth),
            "pred": "",
        }
    except Exception as e:
        print(f"[mopd_reward] Error for data_source={data_source}: {e}")
        traceback.print_exc()
        raise
