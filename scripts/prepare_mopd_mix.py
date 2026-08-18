#!/usr/bin/env python3
"""Build sharded math+code parquets for MOPD multi-teacher experiments.

Tags ability=math|code for sample-level teacher routing.

Writes a directory (not one giant mix file). Code hidden-tests in
reward_model.ground_truth can exceed Arrow's ~2GB `string` limit, so code is
split into shards. Training still concatenates all shards then RandomSamples
across the full set — shard boundaries are not batch boundaries.

Default: balance math:code to 0.5:0.5 by downsampling the larger domain
(limited by the scarcer side). Use --no-balance for raw full concat.

Default sources (G-OPD release):
  https://huggingface.co/datasets/Keven16/G-OPD-Training-Data
  DeepMath-103K/train_filtered_level6.parquet
  Eurus/code_train.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


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
    max_total = min(n_math / math_ratio, n_code / code_ratio)
    target_math = int(max_total * math_ratio)
    target_code = int(max_total * code_ratio)
    target_math = min(target_math, n_math)
    target_code = min(target_code, n_code)
    return target_math, target_code


def _as_out_dir(out: str) -> Path:
    p = Path(out)
    if p.suffix == ".parquet":
        return p.with_suffix("")
    return p


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _shard_code(df: pd.DataFrame, out_dir: Path, shard_rows: int) -> list[Path]:
    if shard_rows <= 0:
        raise SystemExit(f"--code_shard_rows must be > 0, got {shard_rows}")
    if df.empty:
        return []
    n = len(df)
    n_shards = (n + shard_rows - 1) // shard_rows
    width = max(3, len(str(max(n_shards - 1, 0))))
    paths: list[Path] = []
    for i, start in enumerate(range(0, n, shard_rows)):
        chunk = df.iloc[start : start + shard_rows].reset_index(drop=True)
        dest = out_dir / f"code_{i:0{width}d}.parquet"
        _write_parquet(chunk, dest)
        paths.append(dest)
        print(f"  wrote {dest.name} rows={len(chunk)}")
    return paths


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
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/mopd_math_code_mix_balanced",
        help="Output directory (if a .parquet path is given, the suffix is dropped).",
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
    ap.add_argument(
        "--code_shard_rows",
        type=int,
        default=4000,
        help="Max code rows per shard (keeps ground_truth under Arrow's ~2GB string limit).",
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

    for name, df in [("math", math_df), ("code", code_df)]:
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{name} missing columns {missing}; have {list(df.columns)}")

    math_df = math_df[COLS]
    code_df = code_df[COLS]

    out_dir = _as_out_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.parquet"):
        old.unlink()
    manifest = out_dir / "train_files.txt"
    if manifest.exists():
        manifest.unlink()

    written: list[Path] = []
    if len(math_df) > 0:
        math_path = out_dir / "math.parquet"
        _write_parquet(math_df, math_path)
        written.append(math_path)
        print(f"  wrote {math_path.name} rows={len(math_df)}")
    written.extend(_shard_code(code_df, out_dir, args.code_shard_rows))
    if not written:
        raise SystemExit("no rows written (both math and code empty)")

    manifest.write_text("\n".join(str(p.resolve()) for p in written) + "\n")
    hydra_list = "[" + ",".join(json.dumps(str(p.resolve())) for p in written) + "]"
    print(
        f"wrote {out_dir} shards={len(written)} "
        f"math={len(math_df)} code={len(code_df)} "
        f"code_shards={max(len(written) - (1 if math_df.shape[0] else 0), 0)}"
    )
    print(f"manifest {manifest}")
    print(f"hydra data.train_files={hydra_list}")


if __name__ == "__main__":
    main()
