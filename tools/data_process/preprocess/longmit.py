"""Streaming first-stage LongMIT (AR) preprocessing for 28 GB of source data.

Each ``raw/longmit/train.jsonl`` row contains
``{all_docs[{id,content}], clue_docs[{id,content}], question, answer, hop, language,
type}``. It maps to ``memory_docs=[all_docs.content]`` and
``qa=[{question, answer:[answer]}]``, with ``evidence_doc_idx`` resolved by looking
up each ``clue_docs.id`` in ``all_docs``. Rows are yielded incrementally into batched
Parquet writes to keep memory bounded.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_jsonl, write_parquet_stream, row_validator, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402


def iter_rows(limit: int | None = None):
    in_path = os.path.join(RAW_DIR, "longmit", "train.jsonl")
    for i, o in enumerate(read_jsonl(in_path)):
        if limit is not None and i >= limit:
            break
        all_docs = o["all_docs"]
        memory_docs = [d["content"] for d in all_docs]
        clue_ids = {d["id"] for d in o.get("clue_docs") or []}
        evidence = [idx for idx, d in enumerate(all_docs) if d["id"] in clue_ids]
        yield {
            "id": f"longmit-train-{i}",
            "source": "longmit",
            "task_type": "AR",
            "memory_docs": memory_docs,
            "qa": [{
                "question": o["question"],
                "answer": [str(o["answer"])],
                "evidence_doc_idx": evidence,
                "choices": [],
            }],
            "meta": {"n_docs": len(memory_docs), "hop": o.get("hop"),
                     "language": o.get("language"), "type": o.get("type")},
        }


def process_split(limit: int | None = None) -> tuple[str, int]:
    out_path = os.path.join(PROC_DIR, "longmit", "train.parquet")
    n = write_parquet_stream(out_path, iter_rows(limit), validate=row_validator("qa"))
    return out_path, n


def main():
    args = make_arg_parser("LongMIT 第一段预处理").parse_args()
    out_path, n = process_split(args.limit)
    print(f"[longmit] train: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
