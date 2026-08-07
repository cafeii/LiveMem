"""Sample configured per-dataset row counts at load time using a fixed seed.

Training data is fully pre-tokenized under
``dataset/train/<name>/{single_sft,single_rl,pack_rl}.arrow``. At load time, this
module takes the configured number of rows from each dataset and concatenates them
into a mixed table, allowing training volumes to change without regenerating data.

Programmatic usage:
    from mix import load_mixture
    items = [("musique","single_sft",5000), ("narrativeqa","single_sft",None), ...]  # None=all
    table = load_mixture(items, seed=0)          # PyArrow Table
    ds = to_hf(table)                            # Optional datasets.Dataset conversion

CLI sample statistics:
    python tools/data_process/mix.py --spec musique:single_sft:1000,qasper:pack_rl:200
"""

import argparse
import os
import random

import pyarrow as pa

ROOT = os.environ.get("MEMLM_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")


def _load(path: str) -> "pa.Table":
    with pa.ipc.open_file(path) as r:
        return r.read_all()


def sample_table(tbl: "pa.Table", count: int | None, seed: int) -> "pa.Table":
    """Sample ``count`` rows without replacement; return all rows for None or count >= size."""
    n = tbl.num_rows
    if count is None or count >= n:
        return tbl
    idx = sorted(random.Random(seed).sample(range(n), count))
    return tbl.take(idx)


def load_mixture(items, seed: int = 0, shuffle: bool = True) -> "pa.Table":
    """Load, concatenate, and optionally shuffle ``(dataset, split_file, count)`` inputs."""
    tables = []
    for ds, sf, cnt in items:
        path = os.path.join(TRAIN_DIR, ds, f"{sf}.arrow")
        if not os.path.exists(path):
            raise FileNotFoundError(f"训练数据不存在: {path}（先跑 split.py + tokenize_train.py）")
        tables.append(sample_table(_load(path), cnt, seed))
    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    if shuffle:
        idx = list(range(combined.num_rows))
        random.Random(seed).shuffle(idx)
        combined = combined.take(idx)
    return combined


def to_hf(table: "pa.Table"):
    """Convert to a HuggingFace Dataset whose training columns are already native."""
    from datasets import Dataset
    return Dataset(table)


def _parse_spec(spec: str):
    """'musique:single_sft:1000,qasper:pack_rl:200' -> [(ds,sf,count|None)]"""
    items = []
    for part in spec.split(","):
        ds, sf, cnt = part.split(":")
        items.append((ds.strip(), sf.strip(), None if cnt in ("", "all") else int(cnt)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="ds:split_file:count[,...]，count 留空或 all=全取")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = _parse_spec(args.spec)
    tbl = load_mixture(items, seed=args.seed)
    print(f"混合表: {tbl.num_rows} 行, 列={tbl.column_names}")
    lens = [len(x) for x in tbl.column("input_ids").to_pylist()[:2000]]
    if lens:
        import statistics
        print(f"  input_ids 长度(前2000) mean={statistics.mean(lens):.0f} max={max(lens)}")


if __name__ == "__main__":
    main()
