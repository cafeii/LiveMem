"""Convert processed/*_rl.parquet into a balanced mixed verl RLHFDataset Parquet file.

Per request:
  prompt        = [{role: system}, {role: user}]  (chat template applied by verl
                  AgentLoop with add_generation_prompt -> exactly what serve sees)
  data_source   = dataset name (reward dispatch)
  reward_model  = {"style": "rule", "ground_truth": json{kind, questions, golds,
                  judge, group_id}}  (consumed by train/rl/reward.py)
  extra_info    = {index, id, group_id, kind, bucket, max_new, prompt_tokens}

Bucketing/filtering = the shared stride policy (train/sft/eviction_policy.py):
templated prompt token count -> resolve_params / in_range. Out-of-range rows
(prompt <768 or >=61440 templated tokens) are dropped.

Sampling policy:
  * Drop bucket 1 at the recipe level while retaining its serving rule.
  * Include only the 2Wiki pack split.
  * Cap DBpedia and consume its pack split before the single split.
  * Cap bucket 2 and fill remaining capacity with MuSiQue singles.
  * Stream all remaining sources without a per-bucket cap.

Run: /usr/bin/python3 tools/data_process/to_verl_parquet.py [--smoke] [--out DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

WS = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from eval.datasets import DatasetCfg
from eval.prompt import build_requests
from train.sft.eviction_policy import RULE_TABLE, in_range, resolve_params

TOKENIZER = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507")
QUOTAS = {2: 5000}            # hard per-bucket caps; b3/b4/b5 unbounded (except DS_CAP)
DROP_BUCKETS = {1}            # Bucket 1 is excluded from the training mixture.
DS_CAP = {"dbpedia": 1000}    # Per-dataset row cap.
VAL_PER_SOURCE = 11           # balanced val slice: up to N rows per data_source

# Per-dataset recipe.
#   full      stream every split fully (unique-supply / small datasets)
#   cap_first stream splits IN LISTED ORDER until DS_CAP[ds]  (dbpedia: pack first)
#   b2_giant  [pack split full] then [single split streamed until b2 quota full] (musique)
# Sources are ordered by processing cost. MuSiQue remains last so it fills the
# bucket-2 capacity left by earlier sources.
BUILD = [
    ("agnews",          "full",      [("single_rl", "ttl_single", "em"), ("pack_rl", "ttl_multi", "em")]),
    ("qasper",          "full",      [("single_rl", "single", "lm"), ("pack_rl", "multi", "lm")]),
    ("dbpedia",         "cap_first", [("pack_rl", "ttl_multi", "em"), ("single_rl", "ttl_single", "em")]),
    ("narrativeqa",     "full",      [("single_rl", "single", "lm"), ("pack_rl", "multi", "lm")]),
    ("2wikimultihopqa", "full",      [("pack_rl", "multi", "lm")]),
    ("musique",         "b2_giant",  [("pack_rl", "multi", "lm"), ("single_rl", "single", "lm")]),
]


def bucket_of(plen: int) -> int | None:
    """Return the 1-based RULE_TABLE row index, or None when filtered."""
    if not in_range(plen):
        return None
    params = resolve_params(plen)
    return [(cs, w, mn) for _, _, cs, w, mn in RULE_TABLE].index(params) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "dataset" / "train" / "rl_verl"))
    ap.add_argument("--name", default="mix_bal_v1")
    ap.add_argument("--smoke", action="store_true",
                    help="agnews only, no quotas/caps (dev smoke set)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(args.seed)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    taken_b: collections.Counter = collections.Counter()   # per bucket
    taken_ds: collections.Counter = collections.Counter()  # per data_source
    dropped = {"short": 0, "long": 0}
    rows_out: list[dict] = []

    def load(ds: str, split: str) -> list[dict]:
        rows = pq.read_table(WS / "dataset" / "processed" / ds / f"{split}.parquet").to_pylist()
        rng.shuffle(rows)
        return rows

    def consider(cfg: DatasetCfg, row: dict, ds: str, judge: str, ds_cap: int | None = None) -> int:
        """Tokenize one source row's requests, apply quotas/cap, append kept ones."""
        n = 0
        for req in build_requests(cfg, row, reason=True):
            enc = tok.apply_chat_template(req.messages, tokenize=True, add_generation_prompt=True)
            ids = enc["input_ids"] if not isinstance(enc, list) else enc
            plen = len(ids)
            b = bucket_of(plen)
            if b is None:
                dropped["short" if plen < 4096 else "long"] += 1
                continue
            if not args.smoke and b in DROP_BUCKETS:
                dropped["bucket"] = dropped.get("bucket", 0) + 1
                continue
            if not args.smoke and b in QUOTAS and taken_b[b] >= QUOTAS[b]:
                continue
            if ds_cap is not None and taken_ds[ds] >= ds_cap:
                continue
            _, _, _, _, max_new = RULE_TABLE[b - 1]
            gt = json.dumps({"kind": req.kind, "questions": req.questions,
                             "golds": req.golds, "judge": judge, "group_id": req.group_id})
            rows_out.append({
                "prompt": req.messages,
                "data_source": ds,
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {"index": len(rows_out), "id": req.id,
                               "group_id": req.group_id, "kind": req.kind,
                               "bucket": b, "max_new": max_new, "prompt_tokens": plen},
            })
            taken_b[b] += 1
            taken_ds[ds] += 1
            n += 1
        return n

    build = ([("agnews", "full", [("single_rl", "ttl_single", "em"),
                                  ("pack_rl", "ttl_multi", "em")])] if args.smoke else BUILD)

    for ds, mode, splits in build:
        before = taken_ds[ds]
        if mode == "full" or args.smoke:
            for split, kind, judge in splits:
                cfg = DatasetCfg(name=ds, split=split, kind=kind, judge=judge, prompt_name=ds)
                for row in load(ds, split):
                    consider(cfg, row, ds, judge)
        elif mode == "cap_first":
            cap = DS_CAP[ds]
            for split, kind, judge in splits:      # listed order = priority (pack first)
                if taken_ds[ds] >= cap:
                    break
                cfg = DatasetCfg(name=ds, split=split, kind=kind, judge=judge, prompt_name=ds)
                for row in load(ds, split):
                    if taken_ds[ds] >= cap:
                        break
                    consider(cfg, row, ds, judge, ds_cap=cap)
        elif mode == "b2_giant":
            (ps, pk, pj), (ss, sk, sj) = splits
            cfg = DatasetCfg(name=ds, split=ps, kind=pk, judge=pj, prompt_name=ds)   # pack: full
            for row in load(ds, ps):
                consider(cfg, row, ds, pj)
            cfg = DatasetCfg(name=ds, split=ss, kind=sk, judge=sj, prompt_name=ds)   # single: giant
            for row in load(ds, ss):
                if all(taken_b[b] >= q for b, q in QUOTAS.items()):
                    break
                consider(cfg, row, ds, sj)
        print(f"{ds}: +{taken_ds[ds] - before}  buckets={dict(sorted(taken_b.items()))}", flush=True)

    rng.shuffle(rows_out)
    for i, r in enumerate(rows_out):
        r["extra_info"]["index"] = i

    name = "agnews_smoke" if args.smoke else args.name
    df = pd.DataFrame(rows_out)
    df.to_parquet(out_dir / f"{name}.parquet", index=False)

    # Validation is balanced across data sources; limited mode uses the first 64 rows.
    if args.smoke:
        val = df.head(64)
    else:
        by_src: dict[str, list[int]] = collections.defaultdict(list)
        for i in range(len(df)):
            by_src[df.iloc[i]["data_source"]].append(i)
        idx = sorted(i for ii in by_src.values() for i in ii[:VAL_PER_SOURCE])
        val = df.iloc[idx]
    val.to_parquet(out_dir / f"{name}_val.parquet", index=False)

    stats = {"total": len(rows_out),
             "quotas": QUOTAS if not args.smoke else None,
             "ds_cap": DS_CAP if not args.smoke else None,
             "buckets": {str(k): taken_b[k] for k in sorted(taken_b)},
             "data_source": dict(taken_ds), "dropped": dropped, "seed": args.seed}
    (out_dir / f"{name}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {out_dir / (name + '.parquet')} ({len(rows_out)} rows); stats: {stats}", flush=True)


if __name__ == "__main__":
    main()
