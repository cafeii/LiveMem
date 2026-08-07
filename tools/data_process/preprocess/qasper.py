"""First-stage QASPER (AR) preprocessing.

Each row in ``raw/qasper/qasper/{train,validation,test}/0000.parquet`` is one paper:
``{id, title, abstract, full_text{section_name[],paragraphs[][]},
qas{question[],answers[],...}}``. Its one-paper/many-question structure is naturally
packed. ``memory_docs`` contains the full paper (title, abstract, and sections), and
each question becomes one QA item. The first available annotator answer is selected
in priority order: free-form, extractive spans, yes/no, then unanswerable. Train is
processed by default.
"""

import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402


def build_paper(rec: dict) -> str:
    parts = [f"# {rec['title']}", "", rec["abstract"], ""]
    ft = rec["full_text"]
    for name, paras in zip(ft["section_name"], ft["paragraphs"]):
        parts.append(f"## {name}" if name else "##")
        parts.extend(paras)
        parts.append("")
    return "\n".join(parts).strip()


def extract_answer(ann: dict) -> str | None:
    """Return the first usable answer from the annotator list in ``qas['answers'][i]``."""
    for a in ann.get("answer") or []:
        if a.get("free_form_answer"):
            return a["free_form_answer"]
        spans = a.get("extractive_spans")
        if spans:
            return " ".join(spans)
        if a.get("yes_no") is not None:
            return "Yes" if a["yes_no"] else "No"
        if a.get("unanswerable"):
            return "Unanswerable"
    return None


def process_split(split: str, limit: int | None = None) -> tuple[str, int]:
    in_path = os.path.join(RAW_DIR, "qasper", "qasper", split, "0000.parquet")
    records = pq.read_table(in_path).to_pylist()
    if limit is not None:
        records = records[:limit]
    rows = []
    for i, rec in enumerate(records):
        qas = rec["qas"]
        qa = []
        for q, ann in zip(qas["question"], qas["answers"]):
            ans = extract_answer(ann)
            if ans is None:
                continue
            qa.append({"question": str(q), "answer": [ans], "evidence_doc_idx": [], "choices": []})
        if not qa:
            continue
        rows.append({
            "id": f"qasper-{split}-{i}",
            "source": "qasper",
            "task_type": "AR",
            "memory_docs": [build_paper(rec)],
            "qa": qa,
            "meta": {"orig_id": rec["id"], "n_docs": 1, "n_questions": len(qa)},
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, "qasper", f"{split}.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    ap = make_arg_parser("QASPER 第一段预处理")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = ap.parse_args()
    out_path, n = process_split(args.split, args.limit)
    print(f"[qasper] {args.split}: {n} 论文 -> {out_path}")


if __name__ == "__main__":
    main()
