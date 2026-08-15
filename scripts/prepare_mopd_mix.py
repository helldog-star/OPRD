#!/usr/bin/env python3
"""Build a mixed math+code parquet for MOPD multi-teacher experiments.

Tags ability=math|code for sample-level teacher routing.

Default: balance math:code to 0.5:0.5 by downsampling the larger domain
(limited by the scarcer side). Use --no-balance for raw full concat.

Default sources (G-OPD release):
  https://huggingface.co/datasets/Keven16/G-OPD-Training-Data
    DeepMath-103K/train_filtered_level6.parquet
    Eurus/code_train.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n < 0 or n >= len(df):
        return df.reset_index(drop=True)
    if n == 0:
        return df.iloc[0:0].reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _balance_counts(n_math: int, n_code: int, math_ratio: float) -> tuple[int, int]:
    """Max equal-ratio counts under available sizes. math_ratio in (0, 1)."""
    if not (0.0 < math_ratio < 1.0):
        raise SystemExit(f"--math_ratio must be in (0, 1), got {math_ratio}")
    code_ratio = 1.0 - math_ratio
    # Largest total T with floor(T*r) <= n_math and floor(T*(1-r)) <= n_code
    max_total = min(n_math / math_ratio, n_code / code_ratio)
    target_math = int(max_total * math_ratio)
    target_code = int(max_total * code_ratio)
    target_math = min(target_math, n_math)
    target_code = min(target_code, n_code)
    return target_math, target_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--math",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/G-OPD-Training-Data/DeepMath-103K/train_filtered_level6.parquet",
        help="DeepMath filtered parquet (ability will be set to math)",
    )
    ap.add_argument(
        "--code",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/G-OPD-Training-Data/Eurus/code_train.parquet",
        help="Eurus code_train.parquet (ability will be set to code)",
    )
    ap.add_argument(
        "--n_math",
        type=int,
        default=-1,
        help="Subsample math rows before balance; -1 = use all",
    )
    ap.add_argument(
        "--n_code",
        type=int,
        default=-1,
        help="Subsample code rows before balance; -1 = use all",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/mopd_math_code_mix_balanced.parquet",
    )
    ap.add_argument(
        "--math_ratio",
        type=float,
        default=0.5,
        help="Target math share when balancing (default 0.5 → 1:1 math:code).",
    )
    ap.add_argument(
        "--balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance math:code to --math_ratio (default: on). Use --no-balance for full concat.",
    )
    args = ap.parse_args()

    for path, name in [(args.math, "math"), (args.code, "code")]:
        if not Path(path).is_file():
            raise SystemExit(
                f"{name} parquet missing: {path}\n"
                "Download first, e.g.\n"
                "  huggingface-cli download Keven16/G-OPD-Training-Data "
                "--repo-type dataset --local-dir $DATA_DIR/G-OPD-Training-Data"
            )

    math_df = pd.read_parquet(args.math)
    code_df = pd.read_parquet(args.code)
    math_df = _subsample(math_df, args.n_math, args.seed)
    code_df = _subsample(code_df, args.n_code, args.seed + 1)

    if args.balance and args.n_math < 0 and args.n_code < 0:
        target_math, target_code = _balance_counts(len(math_df), len(code_df), args.math_ratio)
        math_df = _subsample(math_df, target_math, args.seed)
        code_df = _subsample(code_df, target_code, args.seed + 1)

    math_df = math_df.copy()
    code_df = code_df.copy()
    math_df["ability"] = "math"
    code_df["ability"] = "code"

    cols = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    for name, df in [("math", math_df), ("code", code_df)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"{name} missing columns {missing}; have {list(df.columns)}")

    mix = pd.concat([math_df[cols], code_df[cols]], ignore_index=True)
    mix = mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mix.to_parquet(out, index=False)
    print(
        f"wrote {out} rows={len(mix)} "
        f"math={(mix.ability == 'math').sum()} code={(mix.ability == 'code').sum()}"
    )


if __name__ == "__main__":
    main()
