"""Tokenize processed splits into ``train/<name>/*.arrow``.

Read the memory-format files
``processed/<name>/{single_sft,single_rl,pack_rl}.parquet``, apply instructions from
``format.py``, pre-tokenize them, add chunk boundaries, loss masks, and dataset
eviction parameters, then write Arrow IPC files. During training, the collator builds
the actual eviction mask and RNN gates from chunk boundaries and limits.

Memory is enclosed in one user turn,
``<|im_start|>user\n{mem}\n\n{instr+Q}<|im_end|>``, followed by the assistant answer.
The collator injects the system segment, so it is not stored here. Loss applies only
to the assistant turn. Chunking has two modes: unit mode groups N passages/items and
puts the QA tail in a separate chunk; token mode splits by token count, appending the
QA tail to the final chunk or starting another if it overflows.

Usage: ``python tools/data_process/tokenize_train.py [--only musique,...] [--limit N]``
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys

import pyarrow as pa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format as F  # noqa: E402
from common import read_parquet, PROC_DIR  # noqa: E402

ROOT = os.environ.get("MEMLM_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507")
MAX_LEN = 128_000  # Drop training samples longer than this limit.

LONG = {"longalign", "longalpaca", "ldc", "longmit"}     # Open-format data.
TTL = {"agnews", "dbpedia"}                              # Numeric-label classification.
SPLIT_FILES = ["single_sft", "single_rl", "pack_rl"]     # Training split sources.

# Per-dataset chunk/eviction settings. unit groups N passages/items; token splits by tokens.
CHUNK_CONFIG = {
    "musique":          {"mode": "unit",  "size": 1,    "chunk_limit": 6, "token_limit": None},
    "2wikimultihopqa":  {"mode": "unit",  "size": 1,    "chunk_limit": 4, "token_limit": None},
    "longmit":          {"mode": "unit",  "size": 1,    "chunk_limit": 8, "token_limit": 16384},
    "agnews":           {"mode": "unit",  "size": 10,   "chunk_limit": 4, "token_limit": None},
    "dbpedia":          {"mode": "unit",  "size": 20,   "chunk_limit": 4, "token_limit": 16384},
    "narrativeqa":      {"mode": "token", "size": 4096, "chunk_limit": 4, "token_limit": None},
    "qasper":           {"mode": "token", "size": 512,  "chunk_limit": 4, "token_limit": None},
    "longalpaca":       {"mode": "token", "size": 1024, "chunk_limit": 3, "token_limit": None},
    "longalign":        {"mode": "token", "size": 1024, "chunk_limit": 4, "token_limit": None},
    "ldc":              {"mode": "token", "size": 1024, "chunk_limit": 3, "token_limit": None},
    "musique-pack":     {"mode": "unit",  "size": 4,    "chunk_limit": 4, "token_limit": None},
    "2wikimultihopqa-pack": {"mode": "unit", "size": 5, "chunk_limit": 4, "token_limit": None},
}

_TOK = None


def get_tok():
    global _TOK
    if _TOK is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return _TOK


def _enc(text: str) -> list[int]:
    return get_tok()(text, add_special_tokens=False)["input_ids"]


def _ans(qa):
    return qa["answer"][0] if qa.get("answer") else ""


# --------------------------------------------------------------------------- #
# Multiprocessing: each worker owns a tokenizer and parallelizes CPU-bound build_sample by row.
# --------------------------------------------------------------------------- #
def _worker_init():
    get_tok()  # Warm the tokenizer in each worker process.


def _build_one(task):
    """Map ``(memory_docs, user_text, assistant_text, cfg)`` to a sample dict or None."""
    memory_docs, user_text, assistant_text, cfg = task
    return build_sample(memory_docs, user_text, assistant_text, cfg)


def _cfg(name, fname):
    if fname == "pack_rl" and f"{name}-pack" in CHUNK_CONFIG:   # Dedicated synthetic-pack settings.
        return CHUNK_CONFIG[f"{name}-pack"]
    return CHUNK_CONFIG[name]


# --------------------------------------------------------------------------- #
# Build (user, assistant) text via format.py based on dataset and single/multi-question mode.
# --------------------------------------------------------------------------- #
def make_text(name, kind, row, reason=True):
    """Disable reasoning for SFT without CoT gold; enable free reasoning for RL."""
    qa = row["qa"]
    if name in LONG:                                            # Open natural chat with no reasoning directive.
        return F.fmt_open(qa[0]["question"], _ans(qa[0]))
    if name in TTL:
        nums = row["meta"]["label_set_nums"]
        if kind == "single":
            return F.fmt_ttl_single(name, qa[0]["question"], _ans(qa[0]), nums, reason)
        return F.fmt_ttl(name, [q["question"] for q in qa], [_ans(q) for q in qa], nums, reason)
    if kind == "single":                                        # Single-question AR.
        return F.fmt_single(name, qa[0]["question"], _ans(qa[0]), reason)
    return F.fmt_multi(name, [q["question"] for q in qa], [_ans(q) for q in qa], reason)  # Multi-question AR/pack.


# --------------------------------------------------------------------------- #
# Sequence layout: memory is enclosed in one user turn and the query follows in that turn:
#   [<|im_start|>user\n{memory chunks}\n\n{instr+Q}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>]
# The collator prepends and retains the system segment; eviction/SWA applies to memory chunks and QA.
# --------------------------------------------------------------------------- #
USER_OPEN = "<|im_start|>user\n"          # Attach the user-turn opener to the first memory chunk.


def _qa_tail(user_text, assistant_text):
    """Append the query, close the user turn, and add the assistant prompt and answer.

    Memory is already inside the user turn. Loss covers only the assistant turn,
    including the ``text`` block and ``<|im_end|>``. A ``\\n\\n`` prefix separates
    memory from the query.
    """
    pre = f"\n\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    post = f"{assistant_text}<|im_end|>"
    pre_ids, post_ids = _enc(pre), _enc(post)
    return pre_ids + post_ids, [0] * len(pre_ids) + [1] * len(post_ids)


def build_sample(memory_docs, user_text, assistant_text, cfg):
    """Build one mechanism-independent training sample, or None above MAX_LEN."""
    qa_ids, qa_loss = _qa_tail(user_text, assistant_text)
    size, mode, tl = cfg["size"], cfg["mode"], cfg.get("token_limit")
    chunks, losses = [], []

    if mode == "unit":
        for i in range(0, len(memory_docs), size):
            seg = "\n".join(memory_docs[i:i + size])
            if i == 0:
                seg = USER_OPEN + seg                           # Put the user-turn opener on the first memory chunk.
            ids = _enc(seg)
            if tl and len(ids) > tl:                            # Re-split oversized unit chunks at token_limit / 2.
                half = tl // 2
                for j in range(0, len(ids), half):
                    chunks.append(ids[j:j + half])
                    losses.append([0] * len(ids[j:j + half]))
            else:
                chunks.append(ids)
                losses.append([0] * len(ids))
        chunks.append(qa_ids)                                   # Keep the query/assistant QA tail in its own chunk.
        losses.append(qa_loss)
        qa_chunk_idx = [len(chunks) - 1]
    else:  # Token mode: split memory plus QA tail continuously; overflow starts a new chunk.
        mem_ids = _enc(USER_OPEN + "\n\n".join(memory_docs))
        full, full_loss = mem_ids + qa_ids, [0] * len(mem_ids) + qa_loss
        for j in range(0, len(full), size):
            chunks.append(full[j:j + size])
            losses.append(full_loss[j:j + size])
        qa_chunk_idx = list(range(len(mem_ids) // size, len(chunks)))

    input_ids = [t for c in chunks for t in c]
    if len(input_ids) > MAX_LEN:
        return None
    loss_mask = [m for ls in losses for m in ls]
    spans, pos = [], 0
    for c in chunks:
        spans.append([pos, pos + len(c)])
        pos += len(c)
    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "chunk_spans": spans,
        "qa_chunk_idx": qa_chunk_idx,
        "chunk_limit": cfg["chunk_limit"],
        "token_limit": cfg["token_limit"] if cfg["token_limit"] is not None else -1,
        "meta": {"n_chunks": len(chunks), "total_len": len(input_ids), "n_qa_chunks": len(qa_chunk_idx)},
    }


# --------------------------------------------------------------------------- #
# Arrow IPC output: native arrays for IDs/masks/indexes and JSON strings for spans/metadata.
# --------------------------------------------------------------------------- #
_INT32 = {"input_ids", "qa_chunk_idx"}
_INT8 = {"loss_mask"}
_JSON = {"chunk_spans", "meta"}


def _to_table(batch):
    cols = {}
    for k in batch[0]:
        vals = [r.get(k) for r in batch]
        if k in _INT32:
            cols[k] = pa.array(vals, type=pa.list_(pa.int32()))
        elif k in _INT8:
            cols[k] = pa.array(vals, type=pa.list_(pa.int8()))
        elif k in _JSON:
            cols[k] = pa.array([json.dumps(v, ensure_ascii=False) for v in vals])
        else:
            cols[k] = pa.array(vals)
    return pa.table(cols)


def write_arrow(out_path, rows_iter):
    """Write Arrow IPC batches from row dicts, treating None as filtered; return both counts."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = schema = None
    n = filtered = 0
    batch = []

    def flush():
        nonlocal writer, schema
        if not batch:
            return
        tbl = _to_table(batch)
        if writer is None:
            schema = tbl.schema
            writer = pa.ipc.new_file(out_path, schema)
        elif tbl.schema != schema:
            tbl = tbl.cast(schema)
        writer.write_table(tbl)
        batch.clear()

    for s in rows_iter:
        if s is None:
            filtered += 1
            continue
        batch.append(s)
        n += 1
        if len(batch) >= 200:
            flush()
    flush()
    if writer is not None:
        writer.close()
    return n, filtered


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def tokenize_file(name, fname, pool, limit=None):
    """Tokenize one split in the shared pool; ``limit`` caps input rows."""
    src = os.path.join(PROC_DIR, name, f"{fname}.parquet")
    if not os.path.exists(src):
        return None
    cfg = _cfg(name, fname)
    kind = "single" if fname.startswith("single") else "pack"
    paradigm = "general" if kind == "single" else "pack"
    tt = "TTL" if name in TTL else "AR"       # Training data contains only AR/TTL.
    reason = not fname.endswith("sft")        # Disable reasoning for SFT; enable it for RL splits.
    out_path = os.path.join(TRAIN_DIR, name, f"{fname}.arrow")

    def tasks():                              # The main process reads Parquet and builds instructions.
        it = read_parquet(src)
        if limit:
            it = itertools.islice(it, limit)
        for row in it:
            u, a = make_text(name, kind, row, reason)
            yield (row["memory_docs"], u, a, cfg)

    def out_rows():                           # Workers build samples in order; the main process assigns IDs.
        i = 0
        for s in pool.imap(_build_one, tasks(), chunksize=16):
            if s is not None:
                s.update(id=f"{name}-{fname}-{i}", source=name, task_type=tt, paradigm=paradigm)
                i += 1
            yield s
    n, filt = write_arrow(out_path, out_rows())
    print(f"[{name}/{fname}] {n} 条 (过滤 {filt}) -> {out_path}", flush=True)
    return n


ALL = ["musique", "2wikimultihopqa", "narrativeqa", "qasper", "agnews", "dbpedia",
       "longalign", "longalpaca", "ldc", "longmit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=None, help="每文件限产 N（debug）")
    ap.add_argument("--nproc", type=int, default=32, help="并行进程数（本机核多内存大）")
    args = ap.parse_args()
    names = [s.strip() for s in args.only.split(",") if s.strip()] or ALL
    # The main process only reads Parquet and applies instructions; tokenizers load in
    # spawned workers. Reuse one spawn pool throughout to avoid repeated setup/teardown.
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.nproc, initializer=_worker_init) as pool:
        for name in names:
            for fname in SPLIT_FILES:
                tokenize_file(name, fname, pool, args.limit)


if __name__ == "__main__":
    main()
