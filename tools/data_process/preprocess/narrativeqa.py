"""First-stage NarrativeQA (AR) preprocessing.

Each row in ``raw/narrativeqa/data/{train,test,validation}-*.parquet`` contains
``{document{id,kind,text(full text),summary,...}, question{text}, answers[{text}]}``.
The source stores one question per row and repeats ``document.text`` across rows.
This preprocessor groups by ``document.id`` into ``memory_docs=[full text]`` and
``qa=[all questions for that document]`` to avoid duplicating documents as large as
800 KB. Train is processed by default.
"""

import glob
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "narrativeqa/data")


def process_split(split: str, limit: int | None = None) -> tuple[str, int]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"{split}-*.parquet")))
    assert files, f"split {split} 无 parquet"
    # Map doc_id to text/kind/QA while preserving insertion order with a dictionary.
    # The limit counts documents: after reaching it, accept questions only for
    # already-recorded documents and do not add new documents.
    docs: dict[str, dict] = {}
    for f in files:
        for rec in pq.read_table(f).to_pylist():
            d = rec["document"]
            did = d["id"]
            if did not in docs:
                if limit is not None and len(docs) >= limit:
                    continue
                docs[did] = {"text": d["text"], "kind": d.get("kind"), "qa": []}
            docs[did]["qa"].append({
                "question": rec["question"]["text"],
                "answer": [a["text"] for a in rec["answers"]],
                "evidence_doc_idx": [0],
                "choices": [],
            })
    rows = []
    for i, (did, info) in enumerate(docs.items()):
        rows.append({
            "id": f"narrativeqa-{split}-{i}",
            "source": "narrativeqa",
            "task_type": "AR",
            "memory_docs": [info["text"]],
            "qa": info["qa"],
            "meta": {"orig_id": did, "n_docs": 1, "kind": info["kind"], "n_questions": len(info["qa"])},
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "narrativeqa", f"{split}.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("NarrativeQA 第一段预处理（按文档聚合）")
    ap.add_argument("--split", choices=["train", "test", "validation"], default="train")
    args = ap.parse_args()
    out_path, n = process_split(args.split, args.limit)
    print(f"[narrativeqa] {args.split}: {n} 文档 -> {out_path}")


if __name__ == "__main__":
    main()
