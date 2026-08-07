#!/usr/bin/env python3
"""Materialize fixed wiki eval subsets under dataset/processed.

The eval stack consumes parquet splits from dataset/processed/<dataset>/.
This script keeps the original full splits untouched and writes sampled split
files next to them so launchers can select the subset by dataset key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data_process"))

from tools.data_process.common import PROC_DIR, read_parquet, write_parquet


WIKI_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")


def _sample_indices(n: int, limit: int, seed: int) -> list[int]:
    if limit >= n:
        return list(range(n))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), limit))


def _sha1_ints(xs: list[int]) -> str:
    payload = ",".join(str(x) for x in xs).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _materialize_one(dataset: str, split: str, limit: int, seed: int) -> dict:
    src = Path(PROC_DIR) / dataset / f"{split}.parquet"
    suffix = f"sample{limit}_seed{seed}"
    dst = Path(PROC_DIR) / dataset / f"{split}_{suffix}.parquet"
    rows = list(read_parquet(str(src)))
    indices = _sample_indices(len(rows), limit, seed)
    sampled = [rows[i] for i in indices]
    n_written = write_parquet(str(dst), sampled)
    if n_written != len(indices):
        raise RuntimeError(f"write count mismatch for {dst}: {n_written} vs {len(indices)}")
    return {
        "dataset": dataset,
        "source_split": split,
        "sample_split": f"{split}_{suffix}",
        "source_path": str(src),
        "sample_path": str(dst),
        "source_rows": len(rows),
        "sample_rows": n_written,
        "limit": limit,
        "seed": seed,
        "indices_sha1": _sha1_ints(indices),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-limit", type=int, default=2000)
    ap.add_argument("--pack-limit", type=int, default=400)
    args = ap.parse_args()

    records = []
    for dataset in WIKI_DATASETS:
        records.append(_materialize_one(dataset, "test_single", args.single_limit, args.seed))
        records.append(_materialize_one(dataset, "test_pack", args.pack_limit, args.seed))

    manifest = {
        "kind": "wiki_eval_sample",
        "seed": args.seed,
        "single_limit": args.single_limit,
        "pack_limit": args.pack_limit,
        "records": records,
    }
    out = Path(PROC_DIR) / f"wiki_eval_sample_seed{args.seed}_manifest.json"
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"manifest={out}")
    for r in records:
        print(
            f"{r['dataset']:<16} {r['sample_split']:<30} "
            f"{r['sample_rows']:>5}/{r['source_rows']:<5} {r['sample_path']}"
        )


if __name__ == "__main__":
    main()
