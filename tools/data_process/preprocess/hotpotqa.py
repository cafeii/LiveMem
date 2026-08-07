"""First-stage HotpotQA (AR) preprocessing.

``raw/hotpotqa/distractor/{validation}-*.parquet`` follows the 2Wiki schema:
``{id, question, answer, type, level, supporting_facts{title,sent_id},
context{title[10],sentences[10][]}}``. Evidence is mixed among ten documents. As in
2Wiki, each document concatenates its title and sentences, and
``evidence_doc_idx`` is resolved from ``supporting_facts.title``. Only the train and
validation splits of the distractor configuration have answers; test exists only in
unlabeled fullwiki, so evaluation uses validation.
"""

import glob
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "hotpotqa/distractor")


def process_split(split: str, limit: int | None = None) -> tuple[str, int]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet")))
    assert files, f"distractor split {split} 无 parquet"
    rows = []
    idx = 0
    for f in files:
        if limit is not None and idx >= limit:
            break
        for rec in pq.read_table(f).to_pylist():
            if limit is not None and idx >= limit:
                break
            ctx = rec["context"]
            docs = [f"{t}\n{' '.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
            sup_titles = set(rec["supporting_facts"]["title"])
            evidence = [i for i, t in enumerate(ctx["title"]) if t in sup_titles]
            rows.append({
                "id": f"hotpotqa-{split}-{idx}",
                "source": "hotpotqa",
                "task_type": "AR",
                "memory_docs": docs,
                "qa": [{
                    "question": str(rec["question"]),
                    "answer": [str(rec["answer"])],
                    "evidence_doc_idx": evidence,
                    "choices": [],
                }],
                "meta": {"orig_id": rec["id"], "n_docs": len(docs),
                         "type": rec.get("type"), "level": rec.get("level")},
            })
            idx += 1
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "hotpotqa", "test.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("HotpotQA(distractor) 第一段预处理")
    ap.add_argument("--split", default="validation", help="distractor 带答案的 split（默认 validation）")
    args = ap.parse_args()
    out_path, n = process_split(args.split, args.limit)
    print(f"[hotpotqa] {args.split} -> test: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
