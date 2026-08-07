"""Generate ``test_single`` and ``test_pack`` evaluation splits under ``processed``.

Datasets fall into four categories based on memory sharing:

- WIKI concatenates different memories into packs: groups of 5 for musique and 10
  for 2wiki/hotpotqa. ``test_single`` preserves individual questions; ``test_pack``
  concatenates G memories and adjusts evidence offsets.
- REGROUP combines rows that share one memory, currently infbench_qa (351 rows over
  69 books). ``test_single`` preserves individual questions; ``test_pack`` collects
  a book's questions into one row with one memory and no evidence offset.
- EXPLODE converts existing one-memory/multi-question rows into single-question rows
  for mab_eventqa/factconsolidation/ttl and locomo. ``test_single`` copies memory per
  question so the harness can deduplicate encoding; ``test_pack`` preserves the
  existing multi-question rows. Locomo is natively question-level and has no pack.
- SINGLE_ONLY contains one-memory/one-question datasets such as infbench_dialogue and
  longmemeval_s/m; only ``test_single`` is produced unchanged.

Usage: ``python tools/data_process/eval_split.py [--only musique,...] [--limit N]``
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_parquet, write_parquet_stream, validate_qa_row, PROC_DIR  # noqa: E402
from split import _pack_group, _mem_row  # noqa: E402

WIKI_GROUP = {"musique": 5, "2wikimultihopqa": 10, "hotpotqa": 10}
REGROUP = {"infbench_qa_eng"}
EXPLODE = {"mab_eventqa", "mab_factconsolidation", "mab_ttl", "locomo"}
NO_PACK = {"locomo"}                       # Natively question-level; no packed split.
SINGLE_ONLY = {"infbench_dialogue_eng", "longmemeval_s", "longmemeval_m"}
ALL = list(WIKI_GROUP) + list(REGROUP) + list(EXPLODE) + list(SINGLE_ONLY)


def _src(name):
    return os.path.join(PROC_DIR, name, "test.parquet")


def _out(name, fn):
    return os.path.join(PROC_DIR, name, f"{fn}.parquet")


def _memkey(row):
    return hashlib.md5("".join(row["memory_docs"]).encode()).hexdigest()


def _copy_single(name, limit=None):
    """Stream source rows unchanged into test_single without loading large sets at once."""
    def stream():
        for i, r in enumerate(read_parquet(_src(name))):
            yield _mem_row(f"{name}-test_single-{i}", name, r["task_type"],
                           r["memory_docs"], r["qa"], r.get("meta", {}))
            if limit and i + 1 >= limit:
                return
    return write_parquet_stream(_out(name, "test_single"), stream(), validate=validate_qa_row)


def _explode_single(name, limit=None):
    """Create one test_single row per QA item, copying the memory."""
    def stream():
        i = 0
        for r in read_parquet(_src(name)):
            for qa in r["qa"]:
                yield _mem_row(f"{name}-test_single-{i}", name, r["task_type"],
                               r["memory_docs"], [qa], r.get("meta", {}))
                i += 1
                if limit and i >= limit:
                    return
    return write_parquet_stream(_out(name, "test_single"), stream(), validate=validate_qa_row)


def _copy_pack(name, limit=None):
    """Preserve existing multi-question rows as test_pack (MAB is already packed)."""
    def stream():
        for i, r in enumerate(read_parquet(_src(name))):
            yield _mem_row(f"{name}-test_pack-{i}", name, r["task_type"],
                           r["memory_docs"], r["qa"], r.get("meta", {}))
            if limit and i + 1 >= limit:
                return
    return write_parquet_stream(_out(name, "test_pack"), stream(), validate=validate_qa_row)


def _wiki(name, limit=None):
    rows = list(read_parquet(_src(name)))
    n1 = _copy_single(name, limit)
    g = WIKI_GROUP[name]
    packs = []
    for pi in range(len(rows) // g):
        row = _pack_group(rows[pi * g:(pi + 1) * g], name, pi)   # Concatenate memories and offset evidence.
        row["id"] = f"{name}-test_pack-{pi}"
        packs.append(row)
        if limit and len(packs) >= limit:
            break
    n2 = write_parquet_stream(_out(name, "test_pack"), iter(packs), validate=validate_qa_row)
    print(f"[{name}] test_single {n1} / test_pack {n2} (wiki concat g={g})")


def _regroup(name, limit=None):
    """Pack rows sharing one memory without duplicating memory or offsetting evidence."""
    n1 = _copy_single(name, limit)
    groups, order = {}, []
    for r in read_parquet(_src(name)):
        k = _memkey(r)
        if k not in groups:
            groups[k] = {"memory_docs": r["memory_docs"], "qa": [], "task_type": r["task_type"]}
            order.append(k)
        groups[k]["qa"].extend(r["qa"])
    packs = []
    for pi, k in enumerate(order):
        gp = groups[k]
        packs.append(_mem_row(f"{name}-test_pack-{pi}", name, gp["task_type"], gp["memory_docs"],
                              gp["qa"], {"n_docs": len(gp["memory_docs"]), "n_qa": len(gp["qa"])}))
        if limit and len(packs) >= limit:
            break
    n2 = write_parquet_stream(_out(name, "test_pack"), iter(packs), validate=validate_qa_row)
    print(f"[{name}] test_single {n1} / test_pack {n2} (regroup by memory)")


def run(name, limit=None):
    if name in WIKI_GROUP:
        _wiki(name, limit)
    elif name in REGROUP:
        _regroup(name, limit)
    elif name in EXPLODE:
        n1 = _explode_single(name, limit)
        n2 = _copy_pack(name, limit) if name not in NO_PACK else 0
        print(f"[{name}] test_single {n1} / test_pack {n2}{' (locomo 原生逐问, 无 pack)' if name in NO_PACK else ''}")
    elif name in SINGLE_ONLY:
        n1 = _copy_single(name, limit)
        print(f"[{name}] test_single {n1} (单 memory 单问, 无 pack)")
    else:
        print(f"[{name}] 跳过（非评估集）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    names = [s.strip() for s in args.only.split(",") if s.strip()] or ALL
    for name in names:
        run(name, args.limit)


if __name__ == "__main__":
    main()
