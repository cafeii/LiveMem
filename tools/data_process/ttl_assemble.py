"""Assemble AG News and DBpedia pools into ICL episodes for training.

Following the data-processing rules, convert atomic classification pools into
memory-QA episodes with the same ``memory_docs + qa`` shape as AR and numeric labels:

- Randomly divide the test pool into ``group``-question groups. For each group,
  sample ``memory`` labeled examples from train without replacement while including
  at least one example of every label.
- Store each example as one ``f"{text}\\nlabel: {number}"`` memory chunk and build QA
  as ``[{question:text, answer:[str(number)]}]``.
- Store numeric candidates for user input in ``meta.label_set_nums``.

AG News uses 8 questions and 100 memory examples per group; DBpedia uses 16 and 200.
Incomplete final groups are discarded, and a fixed random seed ensures reproducibility.
Output goes to ``dataset/processed/<name>/episodes.parquet``. TTL evaluation uses MAB,
so this utility serves training only.
"""

import argparse
import os
import random
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_parquet_stream, row_validator, PROC_DIR  # noqa: E402

CONFIGS = {
    "agnews": {"group": 8, "memory": 100, "seed": 0},
    "dbpedia": {"group": 16, "memory": 200, "seed": 0},
}


def load_pool(name: str, split: str):
    """Return ``[(text, label)]``."""
    t = pq.read_table(os.path.join(PROC_DIR, name, f"{split}.parquet"), columns=["text", "label"])
    return list(zip(t.column("text").to_pylist(), t.column("label").to_pylist()))


def sample_memory(rng, by_label: dict, n_train: int, train, M: int):
    """Sample M unique examples with every label represented; return their indices."""
    chosen, used = [], set()
    for lab, idxs in by_label.items():       # Select one example per label first.
        i = rng.choice(idxs)
        used.add(i)
        chosen.append(i)
    while len(chosen) < M:                    # Fill to M without replacement.
        i = rng.randrange(n_train)
        if i not in used:
            used.add(i)
            chosen.append(i)
    rng.shuffle(chosen)
    return chosen


def iter_episodes(name: str, cfg: dict, limit: int | None):
    train = load_pool(name, "train")
    test = load_pool(name, "test")
    n_train = len(train)
    by_label: dict = {}
    for i, (_, lab) in enumerate(train):
        by_label.setdefault(lab, []).append(i)
    label_set_nums = sorted(by_label)
    G, M = cfg["group"], cfg["memory"]
    rng = random.Random(cfg["seed"])
    order = list(range(len(test)))
    rng.shuffle(order)
    n_groups = len(order) // G                 # Discard an incomplete final group.
    ep = 0
    for g in range(n_groups):
        q_idx = order[g * G:(g + 1) * G]
        mem_idx = sample_memory(rng, by_label, n_train, train, M)
        memory_docs = [f"{train[i][0]}\nlabel: {train[i][1]}" for i in mem_idx]
        qa = [{"question": test[i][0], "answer": [str(test[i][1])],
               "evidence_doc_idx": [], "choices": []} for i in q_idx]
        yield {
            "id": f"{name}-ttl-{ep}",
            "source": name,
            "task_type": "TTL",
            "memory_docs": memory_docs,
            "qa": qa,
            "meta": {"label_set_nums": label_set_nums, "n_mem": M, "n_qa": G, "group_idx": g},
        }
        ep += 1
        if limit is not None and ep >= limit:
            return


def assemble(name: str, limit: int | None = None) -> tuple[str, int]:
    out_path = os.path.join(PROC_DIR, name, "episodes.parquet")
    n = write_parquet_stream(out_path, iter_episodes(name, CONFIGS[name], limit),
                             batch_size=200, validate=row_validator("qa"))
    return out_path, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔：agnews,dbpedia")
    ap.add_argument("--limit", type=int, default=None, help="只产前 N 个 episode（debug）")
    args = ap.parse_args()
    names = [s.strip() for s in args.only.split(",") if s.strip()] or list(CONFIGS)
    for name in names:
        out_path, n = assemble(name, args.limit)
        print(f"[ttl_assemble] {name}: {n} episodes -> {out_path}")


if __name__ == "__main__":
    main()
