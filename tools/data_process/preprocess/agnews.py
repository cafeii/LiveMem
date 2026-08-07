"""First-stage AG News (TTL) preprocessing into an atomic classification pool.

``raw/agnews/data/{train,test}-00000-of-00001.parquet`` contains four classes and
the columns ``{text, label(int)}``. Label names come from ClassLabel metadata;
example selection, grouping, and packing happen during later assembly. Each output
row is ``{id, source:"agnews", task_type:"TTL", split, text, label_text,
meta:{label_set}}``.
"""

import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, get_classlabel_names, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "agnews/data")


def process_split(split: str, limit: int | None = None) -> tuple[str, int]:
    in_path = os.path.join(DATA_DIR, f"{split}-00000-of-00001.parquet")
    names = get_classlabel_names(in_path, "label")
    records = pq.read_table(in_path).to_pylist()
    if limit is not None:
        records = records[:limit]
    rows = [{
        "id": f"agnews-{split}-{i}",
        "source": "agnews",
        "task_type": "TTL",
        "split": split,
        "text": str(rec["text"]),
        "label": int(rec["label"]),                # Numeric label used by training/evaluation to prevent leakage.
        "label_text": names[int(rec["label"])],     # Reference/display only.
        "meta": {"label_set": names},
    } for i, rec in enumerate(records)]
    validate_rows(rows, "ttl", label_fields=["label_text"])
    out_path = os.path.join(PROC_DIR, "agnews", f"{split}.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    args = make_arg_parser("AG News 第一段预处理").parse_args()
    for split in ("train", "test"):
        out_path, n = process_split(split, args.limit)
        print(f"[agnews] {split}: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
