"""Generate the five RULER NIAH tasks used by the release matrix.

Reproduce NVIDIA/RULER NIAH semantics with 128K-token contexts:

- ``s_niah_2``: one needle in an essay haystack with a seven-digit numeric value.
- ``s_niah_3``: one needle in an essay haystack with a UUID value.
- ``mk_niah_1``: one target needle plus three distractors with different keys;
  query only the target.
- ``mq_niah``: four needles and four questions over the same haystack.
- ``mv_niah``: four values for one key, all of which must be recalled.

The official haystack uses Paul Graham essays, supplied as a directory of ``.txt``
files through ``--haystack-dir``. Without local essay data, ``--haystack noise``
falls back to the repeated noise sentence from RULER S-NIAH-1, as documented by
the protocol.

Output is written to ``processed/ruler_niah/test_128k.parquet`` with
``memory_docs=[haystack containing needles]``, ``qa=[{question, answer:[...]}]``,
``source=ruler_<task>``, and depth-related metadata. Multi-question tasks retain all
questions in one row with the shared memory; evaluation with ``kind=single`` splits
them into requests and aggregates them at group level.

Usage:
  /usr/bin/python3 tools/data_process/preprocess/ruler_niah.py \
      --haystack-dir dataset/raw/pg_essays --n 100 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PROC_DIR, validate_rows, write_parquet  # noqa: E402

CTX_TOKENS = 131072            # 128K tier.
BUDGET = CTX_TOKENS - 2048     # Reserve space for prompt, question, and generation.
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
NEEDLE = "One of the special magic numbers for {key} is: {value}.\n"
Q_ONE = "What is the special magic number for {key} mentioned in the provided text?"
Q_MULTI = ("What are all the special magic numbers for {key} mentioned in the "
           "provided text?")

WORDS = ("apple bear castle door eagle forest garden harbor island jungle kite "
         "lantern meadow needle ocean palace quartz river summit temple umbrella "
         "valley willow zephyr anchor bridge canyon dune ember fjord glacier").split()


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        os.environ.get("MEMLM_TOKENIZER", "/nas/lzc/model/qwen3-4b-instruct-2507"))


def _load_haystack(args) -> str:
    if args.haystack == "noise":
        return NOISE * 200000
    files = sorted(glob.glob(os.path.join(args.haystack_dir, "*.txt")))
    assert files, f"--haystack-dir 无 .txt：{args.haystack_dir}"
    return "\n\n".join(open(f, errors="ignore").read() for f in files)


def _key(rng) -> str:
    return f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(100, 999)}"


def _value(rng, kind: str) -> str:
    return (str(rng.randint(1_000_000, 9_999_999)) if kind == "num"
            else str(uuid.UUID(int=rng.getrandbits(128))))


def _build_context(tok, corpus_ids: list[int], needles: list[str],
                   rng) -> str:
    """Trim essay data to BUDGET tokens and insert needles at uniformly sampled depths."""
    start = rng.randint(0, max(0, len(corpus_ids) - BUDGET - 1))
    ids = corpus_ids[start:start + BUDGET]
    text = tok.decode(ids, skip_special_tokens=True)
    pieces, pos = [], 0
    depths = sorted(rng.uniform(0.05, 0.95) for _ in needles)
    for d, n in zip(depths, needles):
        cut = int(len(text) * d)
        pieces += [text[pos:cut], n]
        pos = cut
    pieces.append(text[pos:])
    return "".join(pieces)


def gen_task(task: str, tok, corpus_ids, n: int, rng) -> list[dict]:
    rows = []
    for i in range(n):
        if task == "s_niah_2":
            k, v = _key(rng), _value(rng, "num")
            needles, qa = [NEEDLE.format(key=k, value=v)], \
                [{"question": Q_ONE.format(key=k), "answer": [v]}]
        elif task == "s_niah_3":
            k, v = _key(rng), _value(rng, "uuid")
            needles, qa = [NEEDLE.format(key=k, value=v)], \
                [{"question": Q_ONE.format(key=k), "answer": [v]}]
        elif task == "mk_niah_1":
            keys = [_key(rng) for _ in range(4)]
            vals = [_value(rng, "num") for _ in range(4)]
            needles = [NEEDLE.format(key=k, value=v) for k, v in zip(keys, vals)]
            rng.shuffle(needles)
            qa = [{"question": Q_ONE.format(key=keys[0]), "answer": [vals[0]]}]
        elif task == "mq_niah":
            keys = [_key(rng) for _ in range(4)]
            vals = [_value(rng, "num") for _ in range(4)]
            needles = [NEEDLE.format(key=k, value=v) for k, v in zip(keys, vals)]
            qa = [{"question": Q_ONE.format(key=k), "answer": [v]}
                  for k, v in zip(keys, vals)]
        elif task == "mv_niah":
            k = _key(rng)
            vals = [_value(rng, "num") for _ in range(4)]
            needles = [NEEDLE.format(key=k, value=v) for v in vals]
            # One question with multiple gold values: the NIAH judge scores recall fraction.
            qa = [{"question": Q_MULTI.format(key=k), "answer": vals}]
        else:
            raise ValueError(task)
        ctx = _build_context(tok, corpus_ids, needles, rng)
        tag = f"ruler_{task.replace('_', '')}"   # s_niah_2 -> ruler_sniah2, matching filter_source.
        rows.append({
            "id": f"ruler-{task}-{i}",
            "source": tag,
            "task_type": "AR",  # Exact retrieval; valid schema values are AR/CR/TTL/REC.
            "memory_docs": [ctx],
            "qa": [{**q, "evidence_doc_idx": [0], "choices": []} for q in qa],
            # Evaluation materialization filters meta.orig_source by filter_source substring.
            "meta": {"orig_source": tag, "n_docs": 1, "task": task,
                     "n_questions": len(qa), "ctx_tokens": CTX_TOKENS},
        })
    return rows


TASKS = ("s_niah_2", "s_niah_3", "mk_niah_1", "mq_niah", "mv_niah")


def main():
    ap = argparse.ArgumentParser(description="RULER NIAH 128k 生成")
    ap.add_argument("--haystack", choices=["essay", "noise"], default="essay")
    ap.add_argument("--haystack-dir", default="dataset/raw/pg_essays")
    ap.add_argument("--n", type=int, default=100, help="每任务样本数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = _tok()
    corpus = _load_haystack(args)
    corpus_ids = tok.encode(corpus, add_special_tokens=False)
    assert len(corpus_ids) > BUDGET, \
        f"haystack 语料太短（{len(corpus_ids)} tok < {BUDGET}），补充 .txt 或换 --haystack noise"
    rng = random.Random(args.seed)
    rows = []
    for t in TASKS:
        rows += gen_task(t, tok, corpus_ids, args.n, rng)
        print(f"[ruler] {t}: {args.n} 行")
    validate_rows(rows, "qa")
    out = os.path.join(PROC_DIR, "ruler_niah", "test_128k.parquet")
    n = write_parquet(out, rows)
    print(f"[ruler] {n} 行 -> {out}（haystack={args.haystack}）")


if __name__ == "__main__":
    main()
