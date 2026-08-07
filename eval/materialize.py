"""Evaluation-data materialization and result aggregation.

This module provides loading and aggregation functions for ``eval.pipeline``.
Run evaluation with ``python -m eval.pipeline``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "data_process"))

from common import PROC_DIR, read_parquet  # noqa: E402

from .datasets import get as get_dataset  # noqa: E402
from .matrix import RunSpec  # noqa: E402
from .prompt import Request, build_requests  # noqa: E402

TOK_PATH = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3-4B-Instruct-2507")
RESERVE = 1024  # Headroom for the system prompt and chat template.

# Compatibility aliases used when reading summaries with earlier profile names.
_LEGACY_PROFILE_NAMES = {
    "A32": "state-32k", "A8": "state-8k",
    "B256": "truncate-256k", "B128": "truncate-128k",
    "B32": "truncate-32k", "B8": "truncate-8k",
}


# --------------------------------------------------------------------------- #
# JSONL reading and memory-group hashing.
# --------------------------------------------------------------------------- #
def memory_hash(req: Request) -> str:
    h = hashlib.sha1()
    for d in req.memory_docs or []:
        h.update(d.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# Generation budgets and pack splitting.
# --------------------------------------------------------------------------- #
def _memory_key(row: dict) -> str:
    h = hashlib.sha1()
    for doc in row["memory_docs"]:
        h.update(doc.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()


def _default_max_new_tokens(cfg) -> int:
    """Return the evaluation budget for reasoning plus the answer.

    Single/non-pack uses 1024 and pack uses 8192. Official-protocol datasets
    use per-source official budgets from `datasets.py max_new`.
    """
    if getattr(cfg, "max_new", None) is not None:
        return cfg.max_new
    return 8192 if cfg.group else 1024


def _split_pack(row: dict, max_q: int | None) -> list[dict]:
    """Split a multi-question row into chunks of at most max_q questions.

    If max_q is unset or the row is already short enough, return the original row.
    This is intentionally split-not-truncate: no questions are discarded.
    """
    if max_q is None:
        return [row]
    qa = row.get("qa", [])
    if len(qa) <= max_q:
        return [row]
    chunks = []
    for start in range(0, len(qa), max_q):
        out = dict(row)
        out["id"] = f"{row['id']}#pack{start // max_q}"
        out["qa"] = qa[start:start + max_q]
        meta = dict(out.get("meta", {}))
        meta["eval_pack_cap"] = max_q
        meta["eval_pack_orig_n_qa"] = len(qa)
        meta["eval_pack_start"] = start
        out["meta"] = meta
        chunks.append(out)
    return chunks


def _dynamic_pack_rows(rows: list[dict], max_q: int | None) -> list[dict]:
    """Regroup exploded test_single rows into packs by memory hash.

    Used for one-memory, many-question datasets such as LoCoMo that have no
    test_pack. Wiki packs do not use this path.
    """
    groups: dict[str, dict] = {}
    for row in rows:
        key = _memory_key(row)
        g = groups.setdefault(key, {
            **row,
            "id": f"{row['source']}-dynpack-{len(groups)}",
            "qa": [],
            "meta": {**row.get("meta", {}), "dynamic_pack": True},
        })
        g["qa"].extend(row["qa"])
    packed = []
    for row in groups.values():
        packed.extend(_split_pack(row, max_q))
    return packed


# --------------------------------------------------------------------------- #
# Materialization, result directories, and model-level aggregation.
# --------------------------------------------------------------------------- #
def materialize_requests(dataset_key: str, max_pack_questions: int,
                         limit: int | None, reason: bool = False) -> list[Request]:
    cfg = get_dataset(dataset_key)
    src = os.path.join(PROC_DIR, cfg.name, f"{cfg.split}.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(f"missing parquet: {src}")
    rows = []
    for row in read_parquet(src):
        if cfg.filter_source and cfg.filter_source not in str(
                row.get("meta", {}).get("orig_source", "")):
            continue
        rows.append(row)
    if cfg.dynamic_pack:
        rows = _dynamic_pack_rows(rows, max_pack_questions)
    else:
        rows = [x for r in rows for x in _split_pack(r, max_pack_questions)]
    reqs = [req for row in rows for req in build_requests(cfg, row, reason=reason)]
    if limit:
        reqs = list(itertools.islice(reqs, limit))
    return reqs


def run_dir_of(out_dir: str, spec: RunSpec) -> str:
    return os.path.join(out_dir, spec.model, spec.profile, spec.dataset)


def write_model_summary(out_dir: str, model: str) -> None:
    """Aggregate per-run summaries into model-level JSON and Markdown summaries."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out_dir, model, "*", "*", "summary.json"))):
        row = json.load(open(p))
        if "profile" not in row and "paradigm" in row:
            row["profile"] = _LEGACY_PROFILE_NAMES.get(row["paradigm"], row["paradigm"])
        rows.append(row)
    if not rows:
        return
    with open(os.path.join(out_dir, model, "summary.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    profiles = sorted({r["profile"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    by = {(r["dataset"], r["profile"]): r for r in rows}
    lines = [f"# {model} 评估汇总", "",
             "> 单元格 = item_acc（组级数据集括注 group_acc）；`-`=矩阵不含该档。", "",
             "| dataset | " + " | ".join(profiles) + " |",
             "|---|" + "---|" * len(profiles)]
    for ds in datasets:
        cells = []
        for p in profiles:
            r = by.get((ds, p))
            if not r:
                cells.append("-")
            elif r["kind"] in {"multi", "ttl_multi", "recsys_multi"}:
                cells.append(f"{r['item_acc']:.3f} ({r['group_acc']:.3f})")
            else:
                cells.append(f"{r['item_acc']:.3f}")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    # Rule-based reference columns for multi-metric reporting: EM and token F1;
    # LoCoMo also includes paper-protocol cat1-4 F1.
    lines += ["", "## 规则族参考（EM / F1；locomo 括注 cat1-4 F1=论文口径）", "",
              "| dataset | " + " | ".join(profiles) + " |",
              "|---|" + "---|" * len(profiles)]
    for ds in datasets:
        cells = []
        for p in profiles:
            r = by.get((ds, p))
            if not r or "item_f1" not in r:
                cells.append("-")
            else:
                cell = f"{r.get('item_acc_em', 0):.3f} / {r['item_f1']:.3f}"
                if "item_f1_cat14" in r:
                    cell += f" (c14 {r['item_f1_cat14']:.3f})"
                cells.append(cell)
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    with open(os.path.join(out_dir, model, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
