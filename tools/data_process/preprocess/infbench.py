"""InfBench (AR) preprocessing.

Only ``longbook_qa_eng.jsonl`` and ``longdialogue_qa_eng.jsonl`` are processed;
other InfBench files are skipped. Each row maps from
``{id, context, input, answer(list), options(list)}`` to
``memory_docs=[context], qa=[{question:input, answer, evidence_doc_idx:[0],
choices:options}]``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import read_jsonl, write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

# (input file, source name, output subdirectory)
TASKS = [
    ("longbook_qa_eng.jsonl", "infbench_qa_eng", "infbench_qa_eng"),
    ("longdialogue_qa_eng.jsonl", "infbench_dialogue_eng", "infbench_dialogue_eng"),
]


def to_list_str(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(a) for a in x]
    return [str(x)]


def process_one(in_name: str, source: str, out_subdir: str, limit: int | None = None) -> tuple[str, int]:
    in_path = os.path.join(RAW_DIR, "infbench", in_name)
    out_path = os.path.join(PROC_DIR, out_subdir, "test.parquet")
    rows = []
    for i, obj in enumerate(read_jsonl(in_path)):
        if limit is not None and i >= limit:
            break
        rows.append({
            "id": f"{source}-{obj.get('id', i)}",
            "source": source,
            "task_type": "AR",
            "memory_docs": [obj["context"]],
            "qa": [{
                "question": obj["input"],
                "answer": to_list_str(obj.get("answer")),
                "evidence_doc_idx": [0],
                "choices": list(obj.get("options") or []),
            }],
            "meta": {"orig_id": obj.get("id", i), "n_docs": 1},
        })
    validate_rows(rows, "qa")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    args = make_arg_parser("infbench 第一段预处理").parse_args()
    for in_name, source, out_subdir in TASKS:
        out_path, n = process_one(in_name, source, out_subdir, args.limit)
        print(f"[infbench] {source}: {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
