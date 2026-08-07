"""MemoryAgentBench (MAB) preprocessing.

Parquet rows are already packed and contain
``{context, questions(array), answers(array of arrays), metadata(dict)}``. Each file
mixes multiple sub-datasets distinguished by ``metadata.source`` and filtered by
prefix:

- Accurate_Retrieval includes ruler_qa1/qa2, eventqa_*, and longmemeval_s*. Only
  sources beginning with ``eventqa`` become ``mab_eventqa`` (AR).
- Conflict_Resolution includes factconsolidation_sh/mh_*. Sources beginning with
  ``factconsolidation`` become ``mab_factconsolidation`` (CR).
- Test_Time_Learning includes icl_banking77/clinc150/nlu/trec_coarse/trec_fine and
  recsys_redial. These are prebuilt ICL/recommendation episodes with numeric labels.
  All six clean sources become ``mab_ttl`` (TTL), remain distinguished by
  ``orig_source``, and use source-specific evaluation metrics (classification
  accuracy or Redial recall@k).
- Long_Range_Understanding is skipped because LRU is not used here.

TTL uses the Banking77, CLINC150, NLU, TREC, and Redial evaluation data from MAB.

Rows map to ``memory_docs=[context]`` and
``qa=[{question:q, answer:a, evidence_doc_idx:[], choices:[]} for q,a in
zip(questions,answers)]``. Each Parquet answer is already a ``list[str]`` and is used
directly without another list wrapper. Metadata passes through after conversion from
NumPy/PyArrow values to native Python types.
"""

import json
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet, validate_rows, to_native, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "memoryagentbench/data")
ENTITY2ID = os.path.join(RAW_DIR, "memoryagentbench/entity2id.json")


def _movie_name(uri: str) -> str:
    """Match MAB extract_movie_name by normalizing a DBpedia URI into a movie name."""
    s = uri.split("/")[-1].replace("_", " ").replace("-", " ").replace(">", " ")
    s = re.sub(r"\([^()]*\)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_id2name() -> dict[str, str]:
    """Convert recommender gold DBpedia entity IDs into official-evaluation movie names."""
    return {str(v): _movie_name(k) for k, v in json.load(open(ENTITY2ID)).items()}

# (Parquet file, source, task_type, output subdirectory, metadata.source prefix; "" keeps all)
TASKS = [
    ("Accurate_Retrieval-00000-of-00001.parquet", "mab_eventqa", "AR", "mab_eventqa", "eventqa"),
    ("Conflict_Resolution-00000-of-00001.parquet", "mab_factconsolidation", "CR", "mab_factconsolidation", "factconsolidation"),
    ("Test_Time_Learning-00000-of-00001.parquet", "mab_ttl", "TTL", "mab_ttl", ""),
]


def to_list_str(a):
    """Stringify existing list answers; wrap scalar answers in a one-element list."""
    if a is None:
        return []
    if isinstance(a, (list, tuple)):
        return [str(x) for x in a]
    return [str(a)]


def process_one(fn, source, task_type, out_subdir, src_prefix, limit=None) -> tuple[str, int]:
    table = pq.read_table(os.path.join(DATA_DIR, fn))
    records = table.to_pylist()
    # Select the target sub-dataset by metadata.source prefix, excluding other mixed-in data.
    records = [r for r in records if str((r.get("metadata") or {}).get("source", "")).startswith(src_prefix)]
    if limit is not None:
        records = records[:limit]
    id2name = None
    rows = []
    for i, rec in enumerate(records):
        meta = to_native(rec.get("metadata") or {})
        is_recsys = str(meta.get("source", "")).startswith("recsys")
        if is_recsys and id2name is None:
            id2name = _load_id2name()
        questions = rec["questions"] or []
        answers = rec["answers"] or []
        qa = []
        for q, a in zip(questions, answers):
            gold = to_list_str(a)
            if is_recsys:  # Convert numeric IDs to names; evaluation matches names absent from memory.
                gold = [id2name[x.strip()] for g in gold for x in g.split(",") if x.strip()]
            qa.append({
                "question": str(q),
                "answer": gold,
                "evidence_doc_idx": [],
                "choices": [],
            })
        rows.append({
            "id": f"{source}-{i}",
            "source": source,
            "task_type": task_type,
            "memory_docs": [rec["context"]],
            "qa": qa,
            "meta": {"orig_source": meta.get("source"), "n_qa": len(qa),
                     "qa_pair_ids": meta.get("qa_pair_ids")},
        })
    validate_rows(rows, "qa")
    out_path = os.path.join(PROC_DIR, out_subdir, "test.parquet")
    n = write_parquet(out_path, rows)
    return out_path, n


def main():
    args = make_arg_parser("memoryagentbench 第一段预处理").parse_args()
    for fn, source, task_type, out_subdir, src_prefix in TASKS:
        out_path, n = process_one(fn, source, task_type, out_subdir, src_prefix, args.limit)
        print(f"[mab] {source} ({task_type}): {n} 行 -> {out_path}")


if __name__ == "__main__":
    main()
