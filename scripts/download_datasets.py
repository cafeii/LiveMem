#!/usr/bin/env python3
"""Download the datasets used by LiveMem.

Usage:
  python scripts/download_datasets.py --list                      # List datasets without downloading
  python scripts/download_datasets.py --backend hf                # Download through Hugging Face
  python scripts/download_datasets.py --backend modelscope        # Download through ModelScope
  python scripts/download_datasets.py --only qasper,musique       # Download selected dataset names
  python scripts/download_datasets.py --group train               # Download one group (train or test)
  python scripts/download_datasets.py --backend hf --proxy http://127.0.0.1:7890

Notes:
  - Downloads are stored under dataset/raw/<name>/.
  - The registry covers datasets available from Hugging Face or ModelScope.
  - InfiniteBench is downloaded from xinrongzhang2022/InfiniteBench.
  - MemoryAgentBench is downloaded from ai-hyz/MemoryAgentBench and includes
    the Banking77, CLINC150, NLU, TREC, and ReDial evaluation data.
  - LoCoMo is read directly from third_party/delta-Mem/data/locomo10.json.
  - Some ModelScope namespaces may differ or be unavailable. If a download
    fails, retry with the other backend or update the corresponding ms_id or
    hf_id in DATASETS below.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# (name, hf_id, ms_id, group). A null ms_id reuses hf_id.
DATASETS = [
    # ---------------- Training sets ----------------
    ("qasper",                "allenai/qasper",                            None, "train"),
    ("narrativeqa",           "deepmind/narrativeqa",                      None, "train"),
    ("musique",               "dgslibisey/MuSiQue",                        None, "train"),
    ("2wikimultihopqa",       "framolfese/2WikiMultihopQA",                None, "train"),
    ("agnews",                "fancyzhx/ag_news",                          None, "train"),
    ("dbpedia",               "fancyzhx/dbpedia_14",                       None, "train"),
    ("longalign",             "zai-org/LongAlign-10k",                     None, "train"),
    ("longalpaca",            "Yukang/LongAlpaca-12k",                     None, "train"),
    ("long-data-collections", "emozilla/Long-Data-Collections-Fine-Tune",  None, "train"),
    ("longmit",               "donmaclean/LongMIT-128K",                   None, "train"),
    # ---------------- Test sets ----------------
    ("longmemeval",           "xiaowu0162/longmemeval-cleaned",            None, "test"),
    ("hotpotqa",              "hotpotqa/hotpot_qa",                        None, "test"),
    ("infbench",              "xinrongzhang2022/InfiniteBench",             None, "test"),
    ("memoryagentbench",      "ai-hyz/MemoryAgentBench",                   None, "test"),
    # MuSiQue and 2WikiMultiHopQA use the same source for training and tests;
    # evaluation selects the test split.
]

RAW_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "raw")


def download_hf(repo_id: str, local_dir: str, proxy: str | None) -> None:
    from huggingface_hub import snapshot_download
    if proxy:
        os.environ.setdefault("HTTP_PROXY", proxy)
        os.environ.setdefault("HTTPS_PROXY", proxy)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)


def download_ms(repo_id: str, local_dir: str) -> None:
    from modelscope import snapshot_download
    snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "modelscope"], default="hf")
    ap.add_argument("--only", default="", help="comma-separated dataset names to download")
    ap.add_argument("--group", choices=["train", "test"], default=None)
    ap.add_argument("--proxy", default=None, help="HTTP proxy for Hugging Face downloads, e.g. http://127.0.0.1:7890")
    ap.add_argument("--list", action="store_true", help="list datasets without downloading")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    targets = [d for d in DATASETS
               if (not only or d[0] in only) and (args.group is None or d[3] == args.group)]

    if args.list:
        for name, hf_id, ms_id, group in DATASETS:
            print(f"[{group:5}] {name:24} hf={hf_id}  ms={ms_id or hf_id}")
        return 0

    os.makedirs(RAW_ROOT, exist_ok=True)
    ok, failed = [], []
    for name, hf_id, ms_id, group in targets:
        local_dir = os.path.join(RAW_ROOT, name)
        repo_id = (ms_id or hf_id) if args.backend == "modelscope" else hf_id
        print(f"==> [{args.backend}] {name}: {repo_id} -> {local_dir}")
        try:
            if args.backend == "modelscope":
                download_ms(repo_id, local_dir)
            else:
                download_hf(repo_id, local_dir, args.proxy)
            ok.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"!! {name} download failed; retry with --backend or update its ID in DATASETS")

    print("\n==== Summary ====")
    print("Succeeded:", ", ".join(ok) or "(none)")
    print("Failed:", ", ".join(failed) or "(none)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
