#!/usr/bin/env python3
"""Build a fixed MOPD validation parquet (schema-aligned, no train-time rebuild).

Default is a *train-time* mix (small): full AIME24 + subsampled MATH-500 + code.
For a full offline eval, pass --math500-max 0 --code-max 0 and a different --out.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import datasets as ds
from datasets import Features, Value


FEATURES = Features(
    {
        "prompt": [{"content": Value("string"), "role": Value("string")}],
        "source": Value("string"),
        "id": Value("string"),
        "data_source": Value("string"),
        "ability": Value("string"),
        "reward_model": {"ground_truth": Value("string"), "style": Value("string")},
        "extra_info": {"index": Value("string"), "split": Value("string")},
    }
)


def _normalize_row(row: dict, fallback_source: str, fallback_id: str) -> dict:
    ei = dict(row.get("extra_info") or {})
    idx = ei.get("index", fallback_id)
    ei["index"] = str(idx)
    ei["split"] = str(ei.get("split") or "test")
    rm = dict(row.get("reward_model") or {})
    prompt = [{"content": str(m["content"]), "role": str(m["role"])} for m in row["prompt"]]
    return {
        "prompt": prompt,
        "source": str(row.get("source") or row.get("data_source") or fallback_source),
        "id": str(row.get("id") or fallback_id),
        "data_source": str(row["data_source"]),
        "ability": str(row.get("ability") or "math"),
        "reward_model": {
            "ground_truth": str(rm.get("ground_truth", "")),
            "style": str(rm.get("style", "rule")),
        },
        "extra_info": ei,
    }


def _take(table: ds.Dataset, n: int, seed: int) -> ds.Dataset:
    if n is None or n <= 0 or n >= len(table):
        return table
    return table.shuffle(seed=seed).select(range(n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--aime",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/test_data/AIME24/test.parquet",
    )
    ap.add_argument(
        "--math500",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/test_data/MATH-500/test.parquet",
    )
    ap.add_argument(
        "--code",
        default="/root/siton-tmp/home/liuxinyu/hf_datasets/Eurus/Eurus/code_validation.parquet",
    )
    ap.add_argument(
        "--out",
        default="/root/siton-tmp/home/liuxinyu/OPRD/datasets/test_data/mopd_val_mix.parquet",
    )
    ap.add_argument("--aime-max", type=int, default=0, help="0=all (AIME24 is already small)")
    ap.add_argument("--math500-max", type=int, default=100, help="0=all")
    ap.add_argument("--code-max", type=int, default=64, help="0=all")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows: list[dict] = []
    specs = [
        (args.aime, "AIME24", args.aime_max),
        (args.math500, "MATH-500", args.math500_max),
        (args.code, "code", args.code_max),
    ]
    for path, tag, n_max in specs:
        table = _take(ds.Dataset.from_parquet(path), n_max, args.seed)
        for i, row in enumerate(table):
            src = tag if tag != "code" else (row.get("data_source") or "codecontests")
            rows.append(_normalize_row(row, str(src), f"{tag}_{i}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dset = ds.Dataset.from_list(rows, features=FEATURES)
    dset.to_parquet(str(out))

    from collections import Counter

    print(
        f"wrote {out} n={len(dset)} "
        f"ability={dict(Counter(dset['ability']))} "
        f"data_source={dict(Counter(dset['data_source']))}"
    )


if __name__ == "__main__":
    main()
