"""Streaming first-stage Long-Data-Collections (AR) preprocessing for 6.7 GB of data.

``raw/long-data-collections/data/train-*.parquet`` has 26 shards with the columns
``{text, prompt, completion}``. Prompts use two formats across different shards:

1. Summarization: ``"...\\n\\nQ: <question>\\nA:"`` with the summary as completion.
2. Retrieval QA: ``"...\\n\\nQuestion: <question>"`` with
   ``"\\nAnswer: ...\\nLong Answer: ..."`` as completion.

Rows map to ``memory_docs=[body before the question marker]`` and
``qa=[{question, answer:[completion]}]``. Shards yield rows incrementally into
batched Parquet writes to keep memory bounded. Samples with neither marker are skipped.
"""

import glob
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import write_parquet_stream, row_validator, make_arg_parser, RAW_DIR, PROC_DIR  # noqa: E402

DATA_DIR = os.path.join(RAW_DIR, "long-data-collections/data")
QMARKS = ("\n\nQuestion:", "\n\nQ:")  # Match the more specific Question marker before Q.


def parse_prompt(prompt: str):
    """Return ``(context, question)``, or None when neither marker is present."""
    for marker in QMARKS:
        idx = prompt.rfind(marker)
        if idx == -1:
            continue
        context = prompt[:idx].strip()
        # Remove the summary prompt's trailing "\nA:"; retrieval prompts omit it.
        question = prompt[idx + len(marker):].rsplit("\nA:", 1)[0].strip()
        if context and question:
            return context, question
    return None


def iter_rows(stats: dict, limit: int | None = None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "train-*.parquet")))
    assert files, "long-data-collections 无 parquet"
    count = 0
    for f in files:
        for rec in pq.read_table(f).to_pylist():
            parsed = parse_prompt(rec["prompt"])
            if parsed is None:
                stats["skipped"] += 1
                continue
            context, question = parsed
            yield {
                "id": f"ldc-train-{count}",
                "source": "long_data_collections",
                "task_type": "AR",
                "memory_docs": [context],
                "qa": [{"question": question, "answer": [str(rec["completion"]).strip()],
                        "evidence_doc_idx": [0], "choices": []}],
                "meta": {"n_docs": 1},
            }
            count += 1
            if limit is not None and count >= limit:
                return


def process_split(limit: int | None = None) -> tuple[str, int, int]:
    stats = {"skipped": 0}
    out_path = os.path.join(PROC_DIR, "ldc", "train.parquet")
    n = write_parquet_stream(out_path, iter_rows(stats, limit), validate=row_validator("qa"))
    return out_path, n, stats["skipped"]


def main():
    args = make_arg_parser("Long-Data-Collections 第一段预处理").parse_args()
    out_path, n, skipped = process_split(args.limit)
    print(f"[ldc] train: {n} 行 (跳过无标记 {skipped} 条) -> {out_path}")


if __name__ == "__main__":
    main()
