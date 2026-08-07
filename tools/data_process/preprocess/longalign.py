"""First-stage LongAlign (AR) preprocessing.

Each row in ``raw/longalign/long.jsonl`` is
``{dataset, id, messages[{role,content}], length}``. The first user message separates
the long body from the question at its final ``"\\n\\n"``; the corresponding
assistant message is the answer. In multi-turn samples, later user/assistant pairs
become additional questions about the same body. Rows map to ``memory_docs=[body]``
and ``qa=[{question, answer} for each turn]``. Samples whose question cannot be
separated are skipped. A small number of Chinese-language data samples are retained.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_jsonl, write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402


def build_qa(msgs: list[dict]):
    """Return ``(memory_docs, qa)``, or None when the messages cannot be parsed."""
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        return None
    context, sep, q0 = msgs[0]["content"].rpartition("\n\n")
    if not sep or not context.strip() or not q0.strip():
        return None
    qa = [{"question": q0.strip(), "answer": [msgs[1]["content"]],
           "evidence_doc_idx": [0], "choices": []}]
    # Process subsequent user-question/assistant-answer pairs.
    i = 2
    while i + 1 < len(msgs):
        u, a = msgs[i], msgs[i + 1]
        if u["role"] == "user" and a["role"] == "assistant" and u["content"].strip():
            qa.append({"question": u["content"].strip(), "answer": [a["content"]],
                       "evidence_doc_idx": [0], "choices": []})
        i += 2
    return [context.strip()], qa


def process_split(limit: int | None = None) -> tuple[str, int, int]:
    in_path = os.path.join(RAW_DIR, "longalign", "long.jsonl")
    rows = []
    skipped = 0
    for o in read_jsonl(in_path):
        parsed = build_qa(o["messages"])
        if parsed is None:
            skipped += 1
            continue
        memory_docs, qa = parsed
        rows.append({
            "id": f"longalign-train-{len(rows)}",
            "source": "longalign",
            "task_type": "AR",
            "memory_docs": memory_docs,
            "qa": qa,
            "meta": {"orig_id": o.get("id"), "n_docs": 1, "length": o.get("length")},
        })
        if limit is not None and len(rows) >= limit:
            break
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "longalign", "train.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n, skipped


def main():
    args = make_arg_parser("LongAlign 第一段预处理").parse_args()
    out_path, n, skipped = process_split(args.limit)
    print(f"[longalign] train: {n} 行 (跳过 {skipped} 条) -> {out_path}")


if __name__ == "__main__":
    main()
