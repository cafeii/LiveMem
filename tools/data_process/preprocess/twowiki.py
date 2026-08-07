"""First-stage 2WikiMultiHopQA (AR) preprocessing.

``raw/2wikimultihopqa/data/{train,validation,test}-*.parquet`` contains
``{id, question, answer, type, evidences, supporting_facts{title,sent_id},
context{title[10],sentences[10][]}}``. Each of the ten documents concatenates its
title and sentences. ``evidence_doc_idx`` is resolved from
``supporting_facts.title``, and QA maps to
``[{question, answer:[answer], evidence_doc_idx}]``. Train is processed by default;
the test split is reserved for evaluation.
"""

import glob
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "2wikimultihopqa/data")


def build_docs(context: dict) -> list[str]:
    titles = context["title"]
    sents = context["sentences"]
    return [f"{t}\n{' '.join(s)}" for t, s in zip(titles, sents)]


def process_split(split: str, out_name: str | None = None, limit: int | None = None) -> tuple[str, int]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet")))
    assert files, f"split {split} 无 parquet"
    rows = []
    idx = 0
    for f in files:
        if limit is not None and idx >= limit:
            break
        for rec in pq.read_table(f).to_pylist():
            if limit is not None and idx >= limit:
                break
            ctx = rec["context"]
            docs = build_docs(ctx)
            sup_titles = set(rec["supporting_facts"]["title"])
            evidence = [i for i, t in enumerate(ctx["title"]) if t in sup_titles]
            rows.append({
                "id": f"2wiki-{split}-{idx}",
                "source": "2wikimultihopqa",
                "task_type": "AR",
                "memory_docs": docs,
                "qa": [{
                    "question": str(rec["question"]),
                    "answer": [str(rec["answer"])],
                    "evidence_doc_idx": evidence,
                    "choices": [],
                }],
                "meta": {"orig_id": rec["id"], "n_docs": len(docs), "type": rec.get("type")},
            })
            idx += 1
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "2wikimultihopqa", f"{out_name or split}.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("2WikiMultiHopQA 第一段预处理")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="train")
    ap.add_argument("--out", default=None, help="输出文件名（不含扩展名），默认同 split；"
                    "评估用 validation 时传 test，落到 test.jsonl")
    args = ap.parse_args()
    out_path, n = process_split(args.split, args.out, args.limit)
    print(f"[2wiki] {args.split} -> {os.path.basename(out_path)}: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
