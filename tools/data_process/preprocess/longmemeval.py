"""First-stage LongMemEval (AR) preprocessing.

Each 500-entry ``raw/longmemeval/longmemeval_{s,m}_cleaned.json`` list contains
``{question_id, question_type, question, question_date, answer, answer_session_ids,
haystack_dates, haystack_session_ids, haystack_sessions[[{role,content}...]...]}``.
Each haystack session is rendered as a document with a date header and
``role:content`` turns. ``evidence_doc_idx`` contains the positions of
``answer_session_ids`` in ``haystack_session_ids``, and QA maps to
``[{question, answer:[answer]}]``. Variants s and m are written separately under
``processed/longmemeval_s|m/test.parquet``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

VARIANTS = {"s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json"}


def render_session(turns: list[dict], date: str | None) -> str:
    head = f"[Date: {date}]\n" if date else ""
    body = "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns)
    return head + body


def process_variant(variant: str, limit: int | None = None) -> tuple[str, int]:
    in_path = os.path.join(RAW_DIR, "longmemeval", VARIANTS[variant])
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    rows = []
    for i, o in enumerate(data):
        sessions = o["haystack_sessions"]
        sess_ids = o["haystack_session_ids"]
        dates = o.get("haystack_dates") or [None] * len(sessions)
        memory_docs = [render_session(s, d) for s, d in zip(sessions, dates)]
        ans_ids = set(o.get("answer_session_ids") or [])
        evidence = [j for j, sid in enumerate(sess_ids) if sid in ans_ids]
        rows.append({
            "id": f"longmemeval_{variant}-{o.get('question_id', i)}",
            "source": f"longmemeval_{variant}",
            "task_type": "AR",
            "memory_docs": memory_docs,
            "qa": [{
                "question": o["question"],
                "answer": [str(o["answer"])],
                "evidence_doc_idx": evidence,
                "choices": [],
            }],
            "meta": {"orig_id": o.get("question_id"), "n_docs": len(memory_docs),
                     "question_type": o.get("question_type"), "question_date": o.get("question_date")},
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, f"longmemeval_{variant}", "test.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("LongMemEval 第一段预处理（s + m）")
    ap.add_argument("--variant", choices=["s", "m", "both"], default="both")
    args = ap.parse_args()
    variants = ["s", "m"] if args.variant == "both" else [args.variant]
    for v in variants:
        out_path, n = process_variant(v, args.limit)
        print(f"[longmemeval] {v}: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
