#!/usr/bin/env python3
"""Build a mixed math+code parquet for MOPD multi-teacher pilots.

Keeps verl schema columns and tags ability=math|code for sample-level routing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--math",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/DeepMath-103K/DeepMath-103K/train_filtered_level6.parquet",
    )
    ap.add_argument(
        "--code",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/Eurus/Eurus/code_train.parquet",
    )
    ap.add_argument("--n_math", type=int, default=4000)
    ap.add_argument("--n_code", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/mopd_math_code_mix_8k.parquet",
    )
    args = ap.parse_args()

    math_df = pd.read_parquet(args.math)
    code_df = pd.read_parquet(args.code)
    math_df = _subsample(math_df, args.n_math, args.seed)
    code_df = _subsample(code_df, args.n_code, args.seed + 1)
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
    print(f"wrote {out} rows={len(mix)} math={(mix.ability=='math').sum()} code={(mix.ability=='code').sum()}")


if __name__ == "__main__":
    main()
