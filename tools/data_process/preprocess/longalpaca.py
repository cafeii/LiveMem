"""First-stage LongAlpaca (AR) preprocessing.

``raw/longalpaca/LongAlpaca-12k.json`` is a list of ``{instruction, output}``
entries. About 3,972 of its 12,000 rows contain a long paper and question in this
form: ``"...The paper begins. \n<paper> \n Now the paper ends. \n<question>"``, with
the answer in ``output``. The other roughly 8,028 rows are ordinary short
instructions without a long context for state mode and are skipped. Rows map to
``memory_docs=[paper]`` and ``qa=[{question, answer:[output]}]``.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

BEGIN = "The paper begins."
END = "Now the paper ends."


def parse_paper(instruction: str):
    """Return ``(paper, question)``, or None for non-paper samples."""
    if BEGIN not in instruction or END not in instruction:
        return None
    after = instruction.split(BEGIN, 1)[1]
    paper, _, tail = after.partition(END)
    paper = paper.strip()
    question = tail.strip()
    if question[:9].lower() == "question:":
        question = question[9:].strip()
    if not paper or not question:
        return None
    return paper, question


def process_split(limit: int | None = None) -> tuple[str, int, int]:
    in_path = os.path.join(RAW_DIR, "longalpaca", "LongAlpaca-12k.json")
    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    skipped = 0
    for o in data:
        parsed = parse_paper(o["instruction"])
        if parsed is None:
            skipped += 1
            continue
        paper, question = parsed
        rows.append({
            "id": f"longalpaca-train-{len(rows)}",
            "source": "longalpaca",
            "task_type": "AR",
            "memory_docs": [paper],
            "qa": [{"question": question, "answer": [str(o["output"])],
                    "evidence_doc_idx": [0], "choices": []}],
            "meta": {"n_docs": 1},
        })
        if limit is not None and len(rows) >= limit:
            break
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "longalpaca", "train.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n, skipped


def main():
    args = make_arg_parser("LongAlpaca 第一段预处理").parse_args()
    out_path, n, skipped = process_split(args.limit)
    print(f"[longalpaca] train: {n} 行 (跳过非论文短指令 {skipped} 条) -> {out_path}")


if __name__ == "__main__":
    main()
