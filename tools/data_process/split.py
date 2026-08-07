"""Generate ``processed/<name>/`` splits without altering memory format or adding instructions.

Input is the stage-1 memory format in
``processed/<name>/{train,episodes}.parquet``. Output is written to
``processed/<name>/{single_sft,single_rl,pack_rl}.parquet`` using explicit split
dimensions, a fixed seed, and mutually exclusive assignment.

Split rules use ``seed=0``:

- For single-question-dominant musique/2wiki, group a mutually exclusive 5% of rows
  into ``pack_rl`` (5 per MuSiQue group, 10 per 2Wiki group); divide the other 95%
  one-to-one between ``single_sft`` and ``single_rl``.
- For multi-question-dominant narrativeqa/qasper/agnews/dbpedia, explode 5% of units
  into ``single_sft``, 5% into ``single_rl``, and keep 90% as multi-question
  ``pack_rl`` rows.
- For long-text longalign/longalpaca/ldc/longmit, stream every single-question row
  to ``single_sft`` without shuffling.

Artifacts retain the stage-1 memory schema
``(id/source/task_type/memory_docs/qa/meta)`` and pass ``validate_qa_row``.

Usage: ``python tools/data_process/split.py [--only musique,...] [--limit N]``
"""

import argparse
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_parquet, write_parquet_stream, validate_qa_row, PROC_DIR  # noqa: E402

SEED = 0
PACK_GROUP = {"musique": 5, "2wikimultihopqa": 10}      # Pack sizes for single-dominant data.
MULTI = ["narrativeqa", "qasper", "agnews", "dbpedia"]  # Multi-dominant 5/5/90 unit split.
LONG = ["longalign", "longalpaca", "ldc", "longmit"]    # All rows go to single_sft.
SRC = {"agnews": "episodes", "dbpedia": "episodes"}     # Other datasets default to train.
ALL = list(PACK_GROUP) + MULTI + LONG


def _src_path(name):
    return os.path.join(PROC_DIR, name, f"{SRC.get(name, 'train')}.parquet")


def _mem_row(rid, src, tt, docs, qa, meta):
    return {"id": rid, "source": src, "task_type": tt,
            "memory_docs": docs, "qa": qa, "meta": meta}


def _explode_single(unit, name, side, ui):
    """Explode a multi-question unit into one row per question without changing memory."""
    tt = unit["task_type"]
    for qi, item in enumerate(unit["qa"]):
        yield _mem_row(f"{name}-{side}-{ui}-{qi}", name, tt, unit["memory_docs"], [item],
                       {**unit.get("meta", {}), "from_unit": unit.get("id"), "n_qa": 1})


def _pack_group(group, name, pi):
    """Pack single-question rows by concatenating memory and adjusting evidence offsets."""
    tt = group[0]["task_type"]
    docs, qa, offset = [], [], 0
    for r in group:
        for item in r["qa"]:
            it = dict(item)
            it["evidence_doc_idx"] = [e + offset for e in item.get("evidence_doc_idx", [])]
            qa.append(it)
        docs.extend(r["memory_docs"])
        offset += len(r["memory_docs"])
    return _mem_row(f"{name}-pack-{pi}", name, tt, docs, qa,
                    {"n_docs": len(docs), "n_qa": len(qa), "packed": len(group)})


def split_single_dominant(name, limit=None):
    """Split MuSiQue/2Wiki into 5% exclusive packs and a 95% 1:1 single-row split."""
    rows = list(read_parquet(_src_path(name)))
    if limit:
        rows = rows[:limit * 40]
    random.Random(SEED).shuffle(rows)
    n = len(rows)
    k = int(n * 0.05)
    pack_rows, single_rows = rows[:k], rows[k:]
    h = len(single_rows) // 2
    out = {"single_sft": [], "single_rl": [], "pack_rl": []}
    for i, r in enumerate(single_rows):                          # Split 95% of single rows 1:1.
        r = _mem_row(f"{name}-{'sft' if i < h else 'rl'}-{i}", name, r["task_type"],
                     r["memory_docs"], r["qa"], r.get("meta", {}))
        out["single_sft" if i < h else "single_rl"].append(r)
    g = PACK_GROUP[name]                                          # Group an exclusive 5% into packs.
    for pi in range(len(pack_rows) // g):
        out["pack_rl"].append(_pack_group(pack_rows[pi * g:(pi + 1) * g], name, pi))
    return out


def split_multi_dominant(name, limit=None):
    """Split multi-dominant data into 5% SFT singles, 5% RL singles, and 90% packs."""
    units = list(read_parquet(_src_path(name)))
    if limit:
        units = units[:max(20, limit)]
    random.Random(SEED).shuffle(units)
    n = len(units)
    c5 = max(1, int(n * 0.05))
    out = {"single_sft": [], "single_rl": [], "pack_rl": []}
    for ui, u in enumerate(units[:c5]):
        out["single_sft"].extend(_explode_single(u, name, "sft", ui))
    for ui, u in enumerate(units[c5:2 * c5]):
        out["single_rl"].extend(_explode_single(u, name, "rl", ui))
    for ui, u in enumerate(units[2 * c5:]):                       # Keep 90% of units as packs.
        out["pack_rl"].append(_mem_row(f"{name}-pack-{ui}", name, u["task_type"],
                                       u["memory_docs"], u["qa"], u.get("meta", {})))
    return out


def write_long(name, limit=None):
    """Stream all Long* rows to single_sft without shuffling or full materialization."""
    out_path = os.path.join(PROC_DIR, name, "single_sft.parquet")

    def stream():
        it = read_parquet(_src_path(name))
        it = itertools.islice(it, limit) if limit else it
        for i, r in enumerate(it):
            yield _mem_row(f"{name}-sft-{i}", name, r["task_type"],
                           r["memory_docs"], r["qa"], r.get("meta", {}))
    n = write_parquet_stream(out_path, stream(), validate=validate_qa_row)
    print(f"[{name}/single_sft] {n} 条 -> {out_path}")


def run(name, limit=None):
    if name in LONG:
        write_long(name, limit)
        return
    out = (split_single_dominant if name in PACK_GROUP else split_multi_dominant)(name, limit)
    for fname, rows in out.items():
        if limit:
            rows = rows[:limit]
        path = os.path.join(PROC_DIR, name, f"{fname}.parquet")
        n = write_parquet_stream(path, iter(rows), validate=validate_qa_row)
        print(f"[{name}/{fname}] {n} 条 -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None, help="每文件限产 N（debug）")
    args = ap.parse_args()
    names = [s.strip() for s in args.only.split(",") if s.strip()] or ALL
    for name in names:
        run(name, args.limit)


if __name__ == "__main__":
    main()
