"""First-stage MuSiQue (AR) preprocessing.

Each row in ``raw/musique/musique_ans_v1.0_{train,dev}.jsonl`` contains
``{id, paragraphs[{idx,title,paragraph_text,is_supporting}] x20, question, answer,
answer_aliases, question_decomposition, answerable}``. It maps paragraph text to
``memory_docs``, supporting paragraph positions to ``evidence_doc_idx``, and QA to
``[{question, answer:[primary], ...}]`` while retaining aliases in metadata. By
default, this script processes only train for the first training stage; dev is kept
for evaluation as the test split.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_jsonl, write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

# Split to source filename.
SPLITS = {
    "train": "musique_ans_v1.0_train.jsonl",
    "test": "musique_ans_v1.0_dev.jsonl",  # The official test has no answers; evaluation uses dev.
}


def process_split(split: str, limit: int | None = None) -> tuple[str, int]:
    in_path = os.path.join(RAW_DIR, "musique", SPLITS[split])
    rows = []
    for i, o in enumerate(read_jsonl(in_path)):
        if limit is not None and i >= limit:
            break
        paras = o["paragraphs"]
        memory_docs = [p["paragraph_text"] for p in paras]
        evidence = [idx for idx, p in enumerate(paras) if p.get("is_supporting")]
        rows.append({
            "id": f"musique-{split}-{i}",
            "source": "musique",
            "task_type": "AR",
            "memory_docs": memory_docs,
            "qa": [{
                "question": o["question"],
                "answer": [str(o["answer"])],
                "evidence_doc_idx": evidence,
                "choices": [],
            }],
            "meta": {
                "orig_id": o["id"],
                "n_docs": len(memory_docs),
                "answer_aliases": list(o.get("answer_aliases") or []),
                "hop": o["id"].split("hop")[0] if "hop" in str(o["id"]) else None,
            },
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "musique", f"{split}.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("MuSiQue 第一段预处理")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    args = ap.parse_args()
    out_path, n = process_split(args.split, args.limit)
    print(f"[musique] {args.split}: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
