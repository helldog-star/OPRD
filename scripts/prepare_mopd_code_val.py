#!/usr/bin/env python3
"""Align Eurus code_validation schema with AIME24/MATH-500 for verl concat.

Math vals store extra_info.index as string; Eurus uses int64. datasets.concatenate
refuses to merge those. Also add source/id columns present on math vals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import datasets as ds
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/Eurus/Eurus/code_validation.parquet",
    )
    p.add_argument(
        "--out",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/test_data/code_validation_aligned.parquet",
    )
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    table = ds.Dataset.from_parquet(str(src))
    rows = []
    for i, row in enumerate(table):
        ei = dict(row.get("extra_info") or {})
        idx = ei.get("index", i)
        ei["index"] = str(idx)
        ei.setdefault("split", "test")
        rows.append(
            {
                "prompt": row["prompt"],
                "source": row.get("data_source") or "codecontests",
                "id": f"code_val_{idx}",
                "data_source": row["data_source"],
                "ability": row.get("ability") or "code",
                "reward_model": row["reward_model"],
                "extra_info": ei,
            }
        )

    pdf = pd.DataFrame(rows)
    pdf.to_parquet(out, index=False)
    check = ds.Dataset.from_parquet(str(out))
    print(f"wrote {out} n={len(check)} features={check.features}")


if __name__ == "__main__":
    main()
