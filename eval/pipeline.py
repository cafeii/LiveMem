"""Concurrent multi-profile evaluation pipeline (the sole evaluation entry point).

    python -m eval.pipeline --model delta \\
        --servers '{"truncate":"http://h1:8841/v1,...,http://h2:8846/v1"}' \\
        --judge-url http://h3:8790/v1,http://h3:8791/v1

This replaces the ``run_matrix`` orchestration layer. Evaluation semantics remain
unchanged except for two aligned behaviors: truncation budgets are computed per
group, and judge failures receive a score of zero plus a ``judge_error`` flag.
The pipeline is structured as follows:

    materialize(LPT by descending cost) → tokenizer pool
      (threaded, per-group budget truncation, hash cache)
      → pool_q[tier] → router
      (greedy minimum outstanding below D_high; accounting unit=request)
      → per-server worker
      (HF drains one group sequentially / vLLM runs concurrently after first-item barrier, ≤C)
      → gen_q → evaluation pool
      (inline rules; concurrent round-robin LM judges; failures score zero)
      → per-run records/summary (refresh model summary when a run completes)

- Preserve group affinity: route each group (same memory and truncation) intact to
  one server so snapshot/prefix-cache entries are reused.
- Mix work across datasets: avoid idle time at run boundaries and enqueue large LPT
  groups first to reduce tail latency.
- Bound the backpressure chain: small tok_q/pool_q capacities plus D_high admission
  keep large prompts from accumulating without limit in memory.
- Resume safely: append generations/records one at a time and filter by ID on restart;
  rows containing errors are treated as incomplete and regenerated.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools", "data_process"))

from .datasets import get as get_dataset  # noqa: E402
from .extract import extract  # noqa: E402
from .judge import _JUDGE_PROMPT, aux_scores, judge_answer, locomo_official_f1  # noqa: E402
from .matrix import RunSpec, datasets_for, enumerate_runs  # noqa: E402
from .metrics import aggregate  # noqa: E402
from .materialize import (RESERVE, TOK_PATH, _default_max_new_tokens,  # noqa: E402
                          load_jsonl, materialize_requests, memory_hash,
                          run_dir_of, write_model_summary)
from .prompt import MEM_SEP, Request  # noqa: E402

STATS_PATH = os.path.join(_ROOT, "results", "data_stats", "eval_token_stats.json")

# Methods that use the prefix_chars protocol; the server splits memory/query from user content.
PREFIX_MODELS = {"delta", "c2l", "rag", "recurrent"}
# Single-GPU HF methods drain one group sequentially. RAG/recurrent are GPU-free
# orchestration layers on Qwen3-4B vLLM and use vLLM-style concurrency: the first
# request builds the index/memory block, then sibling requests reuse it.
HF_MODELS = {"delta", "c2l"}


def req_seed(base: int, rid: str) -> int:
    """Derive a stable per-request sampling seed from the global seed and request ID."""
    return int(hashlib.sha1(f"{base}|{rid}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["livemem", "qwen3-4b", "delta", "c2l", "rag", "recurrent"],
                    help="决定评测 profile 与后端类型")
    ap.add_argument("--model-label", default=None,
                    help="结果目录与 summary 的模型名（默认=--model）；"
                         "评新 checkpoint 时可指定独立标签")
    ap.add_argument("--servers", required=True,
                    help='JSON: {"truncate": "url,url,...", '
                         '"state-32k": ..., "state-8k": ...}')
    ap.add_argument("--judge-url", required=True, help="逗号分隔多实例轮询")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--profiles", default=None,
                    help="逗号分隔 profile，如 state-32k,truncate-32k")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results", "eval"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pack-questions", type=int, default=10)
    ap.add_argument("--group-cap", type=int, default=16,
                    help="单组请求数上限：超大组按此拆分成多块分发到不同 server 并行"
                         "（代价=每块所在 server 各自重建一次 prefix/快照）；0=不拆")
    ap.add_argument("--c-vllm", type=int, default=32, help="vllm per-server 发送并发")
    ap.add_argument("--dhigh-vllm", type=int, default=96)
    ap.add_argument("--dhigh-hf", type=int, default=3)
    ap.add_argument("--judge-concurrency", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="采样种子基数；逐请求种子由 seed 与 request id 稳定派生")
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reason", action="store_true",
                    help="prompt 诱导 CoT（评 RL ckpt 用：RL 训练 reason=True 口径）")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class LiteReq:
    """Lightweight scoring request without memory_docs, which bounds pending-run memory."""
    id: str
    kind: str
    questions: list[str]
    golds: list[list[str]]
    group_id: str
    cats: list | None = None


@dataclass
class GroupTask:
    spec: RunSpec
    pool_key: str                       # truncate / state-32k / state-8k
    items: list[tuple]                  # [(req_id, messages, extra)]
    max_new: int


@dataclass
class RunState:
    spec: RunSpec
    judge_type: str
    max_new: int
    rdir: str
    total: int = 0
    lite: dict = field(default_factory=dict)       # id -> LiteReq
    scored: set = field(default_factory=set)
    gen_failed: set = field(default_factory=set)
    n_judge_fail: int = 0
    gen_s0: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def finished(self) -> bool:
        return len(self.scored) + len(self.gen_failed) >= self.total


# --------------------------------------------------------------------------- #
# Rough cost estimates for LPT ordering
# --------------------------------------------------------------------------- #
def load_cost_table() -> dict:
    try:
        with open(STATS_PATH) as f:
            stats = json.load(f)["datasets"]
        return {k: v["n_requests"] * (min(v["prompt_tokens"]["p50"], 262144) +
                                      v["max_new_tokens"])
                for k, v in stats.items() if "prompt_tokens" in v}
    except Exception:  # noqa: BLE001 -- skip cost ordering when statistics are unavailable
        return {}


def order_runs(runs: list[RunSpec]) -> list[RunSpec]:
    cost = load_cost_table()
    return sorted(runs, key=lambda r: cost.get(r.dataset, 0), reverse=True)


# --------------------------------------------------------------------------- #
# Tokenizer pool: materialize → group → truncate to per-group budget → pool_q
# --------------------------------------------------------------------------- #
class Tokenizer:
    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(TOK_PATH)
        self._mem_ids: dict[str, list[int]] = {}
        self._ut_len: dict[str, int] = {}

    def memory_budget_of(self, spec: RunSpec, max_new: int, user_tokens: int) -> int:
        """Return the number of memory tokens that this group may retain.

        State mode processes the full history and is limited only by the model's
        context limit. Truncate mode further restricts history to the most recent
        ``window_size`` tokens. Both reserve space for the query, template, and output.
        """
        hard_budget = spec.context_limit - max_new - RESERVE - user_tokens
        if hard_budget <= 0:
            raise ValueError(
                f"{spec}: user_text {user_tokens} tok + generation {max_new} "
                f"exceeds context limit {spec.context_limit}")
        if spec.mode == "truncate":
            return min(spec.window_size, hard_budget)
        return hard_budget

    def prepare_group(self, spec: RunSpec, reqs: list[Request],
                      max_new: int, is_delta: bool,
                      send_doc_lens: bool = False) -> list[tuple]:
        """Convert one same-memory group into budget-consistent request tuples."""
        mem = MEM_SEP.join(reqs[0].memory_docs)
        ut_max = 0
        for r in reqs:
            uh = hashlib.sha1(r.user_text.encode("utf-8", errors="ignore")).hexdigest()
            if uh not in self._ut_len:
                self._ut_len[uh] = len(self.tok.encode(r.user_text, add_special_tokens=False))
            ut_max = max(ut_max, self._ut_len[uh])
        mem_budget = self.memory_budget_of(spec, max_new, ut_max)
        if len(mem) <= mem_budget:  # Token count cannot exceed character count; no truncation needed.
            out_mem = mem
        else:
            h = hashlib.sha1(mem.encode("utf-8", errors="ignore")).hexdigest()
            if h not in self._mem_ids:
                self._mem_ids[h] = self.tok.encode(mem, add_special_tokens=False)
            ids = self._mem_ids[h]
            out_mem = (mem if len(ids) <= mem_budget else
                       self.tok.decode(ids[-mem_budget:], skip_special_tokens=True))
        # RAG/recurrent split units at the original memory_docs boundaries. When the
        # joined memory is unmodified, pass each document's character length through.
        # Truncation breaks those boundaries, so the server then infers them from ``\n\n``.
        doc_lens = ([len(d) for d in (reqs[0].memory_docs or [])]
                    if send_doc_lens and out_mem is mem else None)
        items = []
        for r in reqs:
            messages = (r.messages if out_mem is mem else
                        r.messages[:-1] + [{**r.messages[-1],
                                            "content": f"{out_mem}{MEM_SEP}{r.user_text}"}])
            extra = None
            if is_delta:
                extra = {"prefix_chars": len(out_mem) + len(MEM_SEP)}
                if doc_lens is not None:
                    extra["doc_lens"] = doc_lens
            items.append((r.id, messages, extra))
        return items


async def producer(pkey: str, runs: list[RunSpec], states: dict, args,
                   pool_q: asyncio.Queue, gen_q: asyncio.Queue, log):
    """Produce one tier's work in LPT order: materialize, group, tokenize, and enqueue.

    Each pool has an independent coroutine so a saturated pool cannot block the others.
    Generated-but-unscored requests go directly to ``gen_q``; completed runs are skipped.
    """
    tk = Tokenizer()
    is_delta = args.model in PREFIX_MODELS  # Use the prefix_chars protocol, including RAG/recurrent.
    send_doc_lens = args.model in ("rag", "recurrent")  # Preserve original document boundaries.
    for spec in runs:
        st: RunState = states[key_of(spec)]
        gen_path = os.path.join(st.rdir, "generations.jsonl")
        done_out: dict[str, str] = {}
        for rec in load_jsonl(gen_path):
            if "output" in rec:
                done_out[rec["request"]["id"]] = rec["output"]

        reqs = await asyncio.to_thread(
            materialize_requests, spec.dataset, args.max_pack_questions, args.limit,
            args.reason)
        st.total = len(reqs)
        for r in reqs:
            st.lite[r.id] = LiteReq(
                r.id, r.kind, r.questions, r.golds, r.group_id, r.cats)

        # Send generated-but-unscored items straight to evaluation. In gen-only mode,
        # skip this because there is no consumer and the bounded queue could deadlock.
        n_rescore = 0
        if not args.gen_only:
            for rid, out in done_out.items():
                if rid in st.lite and rid not in st.scored:
                    await gen_q.put((st, st.lite[rid], out))
                    n_rescore += 1
        todo = [r for r in reqs if r.id not in done_out]
        if st.finished() and not todo:
            if not os.path.exists(os.path.join(st.rdir, "summary.json")):
                await maybe_finish_run(st, args, log)
            continue
        if done_out or st.scored:
            log(f"[resume] {key_of(spec)}: gen 已有 {len(done_out)}, 已打分 {len(st.scored)}, "
                f"补打分 {n_rescore}, 待生成 {len(todo)}")
        if not todo:
            continue

        groups: dict[str, list[Request]] = {}
        for r in todo:
            groups.setdefault(memory_hash(r), []).append(r)
        ordered = sorted(groups.values(), key=len, reverse=True)  # LPT within a run.
        # Split groups larger than group_cap across servers to reduce tail latency
        # (for example, one 200-item movie_rec group). Each destination must rebuild
        # the prefix/snapshot for its chunk.
        if args.group_cap:
            ordered = [g[i:i + args.group_cap] for g in ordered
                       for i in range(0, len(g), args.group_cap)]
        for greqs in ordered:
            items = await asyncio.to_thread(
                tk.prepare_group, spec, greqs, st.max_new, is_delta, send_doc_lens)
            await pool_q.put(GroupTask(spec, pkey, items, st.max_new))
    await pool_q.put(None)  # No more groups for this tier.


# --------------------------------------------------------------------------- #
# Generation pool: Server + per-pool router + per-server worker
# --------------------------------------------------------------------------- #
def make_client(url: str, timeout: float):
    import httpx
    from openai import AsyncOpenAI
    # Disable connection reuse for cross-host network devices that close idle connections.
    return AsyncOpenAI(
        base_url=url, api_key="EMPTY", max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=0)),
            timeout=httpx.Timeout(timeout, connect=15.0)))


class Server:
    def __init__(self, url: str, kind: str, c: int, d_high: int):
        self.url, self.kind, self.c, self.d_high = url, kind, c, d_high
        self.outstanding = 0
        self.q: asyncio.Queue = asyncio.Queue()
        # The timeout must cover both queueing and generation for long prompts at D_high.
        self.client = make_client(url, timeout=1500.0 if kind == "hf" else 1500.0)
        self.model: str | None = None
        self.sem = asyncio.Semaphore(max(1, c))
        self.n_done = 0


class GenEngine:
    def __init__(self, args, states: dict, gen_q: asyncio.Queue, log):
        self.args, self.states, self.gen_q, self.log = args, states, gen_q, log
        kind = "hf" if args.model in HF_MODELS else "vllm"
        self.pools: dict[str, list[Server]] = {}
        for pkey, urls in json.loads(args.servers).items():
            c = 1 if kind == "hf" else args.c_vllm
            dh = args.dhigh_hf if kind == "hf" else args.dhigh_vllm
            self.pools[pkey] = [Server(u.strip(), kind, c, dh)
                                for u in urls.split(",") if u.strip()]
        self.cond = asyncio.Condition()  # Notify routers when outstanding counts change.
        self.retries = 4
        self.gen_kw = dict(temperature=args.temperature, top_p=args.top_p)
        self.extra = {"top_k": args.top_k, "min_p": args.min_p}

    async def router(self, pkey: str, pool_q: asyncio.Queue):
        """Route greedily to the least-loaded server below D_high; wait if all are full."""
        servers = self.pools[pkey]
        while True:
            task = await pool_q.get()
            if task is None:
                for s in servers:
                    await s.q.put(None)
                return
            while True:
                cands = [s for s in servers if s.outstanding < s.d_high]
                if cands:
                    srv = min(cands, key=lambda s: s.outstanding)
                    srv.outstanding += len(task.items)
                    await srv.q.put(task)
                    break
                async with self.cond:
                    await self.cond.wait()

    async def _one(self, srv: Server, st: RunState, rid: str,
                   messages: list[dict], extra: dict | None, max_new: int):
        body = {**self.extra, **(extra or {}), "seed": req_seed(self.args.seed, rid)}
        last = None
        res = None
        for i in range(self.retries):
            try:
                if srv.model is None:
                    srv.model = (await srv.client.models.list()).data[0].id
                async with srv.sem:
                    resp = await srv.client.chat.completions.create(
                        model=srv.model, messages=messages, max_tokens=max_new,
                        extra_body=body, **self.gen_kw)
                res = {"output": resp.choices[0].message.content or ""}
                break
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(min(5 * (i + 1), 30))
        if res is None:
            res = {"error": f"{type(last).__name__}: {last}"}
        async with st.lock:
            with open(os.path.join(st.rdir, "generations.jsonl"), "a") as f:
                f.write(json.dumps(
                    {"request": {"id": rid}, **res}, ensure_ascii=False) + "\n")
        srv.n_done += 1
        srv.outstanding -= 1
        async with self.cond:
            self.cond.notify_all()
        if "error" in res:
            st.gen_failed.add(rid)
            self.log(f"[gen-fail] {key_of(st.spec)} {rid}: {res['error'][:120]}")
            await maybe_finish_run(st, self.args, self.log)
        elif not self.args.gen_only:
            await self.gen_q.put((st, st.lite[rid], res["output"]))

    async def _run_group(self, srv: Server, task: GroupTask):
        st = self.states[key_of(task.spec)]
        first, *rest = task.items
        # First-item barrier: establish the prefix cache/snapshot before sibling requests.
        await self._one(srv, st, first[0], first[1], first[2], task.max_new)
        if rest:
            await asyncio.gather(*[self._one(srv, st, rid, m, e, task.max_new)
                                   for rid, m, e in rest])

    async def worker(self, srv: Server):
        """Process HF groups sequentially and allow concurrent groups with vLLM.

        Sequential draining keeps the HF snapshot LRU of one stable. vLLM must process
        groups concurrently to fill batches when groups contain a single item; ``srv.sem``
        (C) and D_high admission jointly bound concurrency.
        """
        if srv.kind == "hf":
            while True:
                task = await srv.q.get()
                if task is None:
                    return
                st = self.states[key_of(task.spec)]
                for rid, messages, extra in task.items:
                    await self._one(srv, st, rid, messages, extra, task.max_new)
        else:
            group_tasks = []
            while True:
                task = await srv.q.get()
                if task is None:
                    break
                group_tasks.append(asyncio.create_task(self._run_group(srv, task)))
            await asyncio.gather(*group_tasks)


# --------------------------------------------------------------------------- #
# Evaluation pool
# --------------------------------------------------------------------------- #
class Evaluator:
    def __init__(self, args, log):
        self.args, self.log = args, log
        self.clients = [make_client(u.strip(), timeout=120.0)
                        for u in args.judge_url.split(",") if u.strip()]
        self.judge_model = args.judge_model
        self.sem = asyncio.Semaphore(args.judge_concurrency)
        self._rr = 0
        self.retries = 4

    async def _judge_lm(self, q: str, gold: list[str], pred: str) -> tuple[bool, bool]:
        """Return ``(correct, judge_error)``; score zero after all retries fail."""
        self._rr += 1
        client = self.clients[self._rr % len(self.clients)]
        prompt = _JUDGE_PROMPT.format(q=q[:2000], gold=" | ".join(gold)[:1000],
                                      pred=pred[:4000])
        for i in range(self.retries):
            try:
                if self.judge_model is None:
                    self.judge_model = (await client.models.list()).data[0].id
                async with self.sem:
                    resp = await client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=8,
                        # Disable thinking for thinking-family judges (Qwen3.6-35B), or
                        # <think> consumes max_tokens and every result becomes NO.
                        # Non-thinking templates silently ignore this setting.
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                txt = (resp.choices[0].message.content or "").strip().upper()
                return txt.startswith("YES"), False
            except Exception:  # noqa: BLE001
                await asyncio.sleep(min(5 * (i + 1), 30))
        return False, True

    async def score_one(self, st: RunState, req: LiteReq, output: str) -> dict:
        n = len(req.questions)
        parsed = extract(output, req.kind, n_questions=n)
        answers = parsed["answers"]
        items = []
        for i in range(1, n + 1):
            pred = answers.get(i, "")
            q, gold = req.questions[i - 1], req.golds[i - 1]
            item = {"q": q, "gold": gold, "pred": pred}
            if st.judge_type == "lm":
                if pred and gold:
                    item["correct"], jerr = await self._judge_lm(q, gold, pred)
                    if jerr:
                        item["judge_error"] = True
                        st.n_judge_fail += 1
                else:
                    item["correct"] = False
            else:
                item["correct"] = bool(pred) and judge_answer(st.judge_type, q, gold, pred)
            item.update(aux_scores(gold, pred))  # Reference columns for rule-based metrics.
            cat = req.cats[i - 1] if req.cats else None
            if cat is not None:
                item["cat"] = cat
                item["f1_official"] = round(locomo_official_f1(cat, gold, pred), 4)
            items.append(item)
        return {"id": req.id, "group_id": req.group_id,
                "n_questions": n, "n_found": parsed["n_found"],
                "has_block": parsed["has_block"],
                "format_ok": parsed["n_found"] == n,
                "items": items, "raw": output}

    async def worker(self, gen_q: asyncio.Queue):
        while True:
            got = await gen_q.get()
            if got is None:
                return
            st, req, output = got
            rec = await self.score_one(st, req, output)
            async with st.lock:
                if rec["id"] in st.scored:
                    continue
                with open(os.path.join(st.rdir, "records.jsonl"), "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                st.scored.add(rec["id"])
            await maybe_finish_run(st, self.args, self.log)


# --------------------------------------------------------------------------- #
# Run finalization
# --------------------------------------------------------------------------- #
def key_of(spec: RunSpec) -> str:
    return f"{spec.model}/{spec.profile}/{spec.dataset}"


async def maybe_finish_run(st: RunState, args, log):
    if not st.finished() or st.total == 0:
        return
    async with st.lock:
        done_flag = os.path.join(st.rdir, "summary.json")
        records = load_jsonl(os.path.join(st.rdir, "records.jsonl"))
        agg = aggregate(records)  # Aggregate all rule-based reference metrics centrally.
        cfg = get_dataset(st.spec.dataset)
        summary = {
            "run": f"{args.model_label}/{st.spec.profile}/{st.spec.dataset}",
            "model": args.model_label,
            "profile": st.spec.profile, "mode": st.spec.mode,
            "window_size": st.spec.window_size, "dataset": st.spec.dataset,
            "split": cfg.split, "kind": cfg.kind, "judge": st.judge_type,
            "context_limit": st.spec.context_limit,
            "n_requests": len(records), "n_gen_fail": len(st.gen_failed),
            "n_judge_fail": st.n_judge_fail,
            "gen_s": round(time.time() - st.gen_s0, 1),
            "max_new_tokens": st.max_new,
            "generation": {"temperature": args.temperature, "top_p": args.top_p,
                           "top_k": args.top_k, "min_p": args.min_p,
                           "seed": args.seed},
            "max_pack_questions": args.max_pack_questions, "limit": args.limit,
            "pipeline": "plan03.1",
            **agg,
        }
        with open(done_flag, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    write_model_summary(args.out_dir, args.model_label)
    log(f"[summary] {key_of(st.spec)} item_acc={agg['item_acc']:.3f} "
        f"group_acc={agg['group_acc']:.3f} fmt={agg['format_ok_rate']:.3f} "
        f"gen_fail={len(st.gen_failed)} judge_fail={st.n_judge_fail}")


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
async def main_async(args):
    def log(msg):
        print(msg, flush=True)

    default_ds = datasets_for(args.model)
    datasets = ([x.strip() for x in args.datasets.split(",") if x.strip()]
                if args.datasets else default_ds)
    profiles = ([x.strip() for x in args.profiles.split(",") if x.strip()]
                if args.profiles else None)
    runs = order_runs(enumerate_runs(args.model, datasets, profiles))
    server_pools = set(json.loads(args.servers).keys())
    skipped = [r for r in runs if r.server_pool not in server_pools]
    if skipped:
        log(f"[skip] servers 缺 profile，跳过 {len(skipped)} 个 run: "
            f"{sorted({r.profile for r in skipped})}")
        runs = [r for r in runs if r not in skipped]
    log(f"[pipeline] model={args.model} runs={len(runs)} (LPT 成本降序)")
    if args.dry_run:
        for r in runs:
            print(f"  {key_of(r)}")
        return

    states: dict[str, RunState] = {}
    for spec in runs:
        cfg = get_dataset(spec.dataset)
        rdir = os.path.join(args.out_dir, args.model_label, spec.profile, spec.dataset)
        os.makedirs(rdir, exist_ok=True)
        st = RunState(spec, cfg.judge, _default_max_new_tokens(cfg), rdir)
        st.scored = {r["id"] for r in load_jsonl(os.path.join(rdir, "records.jsonl"))}
        states[key_of(spec)] = st

    gen_q: asyncio.Queue = asyncio.Queue(maxsize=2048)
    engine = GenEngine(args, states, gen_q, log)
    pool_qs = {pkey: asyncio.Queue(maxsize=2) for pkey in engine.pools}

    evaluator = Evaluator(args, log) if not args.gen_only else None
    if evaluator:
        # Validate judge availability at startup so an unreachable service cannot turn
        # an entire LM-judged dataset into zero scores.
        for c in evaluator.clients:
            try:
                await asyncio.wait_for(c.models.list(), timeout=10)
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"[abort] judge 不可达: {c.base_url} ({e}); "
                                 f"起 judge 后重跑，或用 --gen-only 先生成") from e
    eval_workers = ([asyncio.create_task(evaluator.worker(gen_q))
                     for _ in range(args.judge_concurrency)] if evaluator else [])

    async def status():
        while True:
            await asyncio.sleep(120)
            done = sum(1 for st in states.values()
                       if st.total and st.finished())
            gen = sum(len(st.scored) + len(st.gen_failed) for st in states.values())
            srv = " ".join(f"{s.url.split('//')[1].split('/')[0]}:{s.outstanding}"
                           for pool in engine.pools.values() for s in pool)
            log(f"[status] runs 完成 {done}/{len(runs)} | 已收口请求 {gen} | "
                f"server outstanding: {srv}")

    status_task = asyncio.create_task(status())

    workers = [asyncio.create_task(engine.worker(s))
               for pool in engine.pools.values() for s in pool]
    routers = [asyncio.create_task(engine.router(pkey, pool_qs[pkey]))
               for pkey in engine.pools]
    # Give each server pool an independent producer so one pool cannot block the others.
    by_pool: dict[str, list[RunSpec]] = {}
    for spec in runs:
        by_pool.setdefault(spec.server_pool, []).append(spec)
    producers = [producer(pk, rs, states, args, pool_qs[pk], gen_q, log)
                 for pk, rs in by_pool.items()]
    # Send termination sentinels for empty tiers so their routers/workers can finish.
    for pk in engine.pools:
        if pk not in by_pool:
            producers.append(pool_qs[pk].put(None))
    await asyncio.gather(*producers)
    await asyncio.gather(*routers)
    await asyncio.gather(*workers)
    if evaluator:
        # Once generation is complete, drain gen_q before stopping evaluators.
        while not gen_q.empty():
            await asyncio.sleep(1)
        for _ in eval_workers:
            await gen_q.put(None)
        await asyncio.gather(*eval_workers)
    status_task.cancel()

    unfinished = [k for k, st in states.items() if st.total and not st.finished()]
    n_gen_fail = sum(len(st.gen_failed) for st in states.values())
    log(f"[all done] {args.out_dir}/{args.model} "
        f"(gen_fail={n_gen_fail}, 未收口 run={len(unfinished)}: {unfinished[:5]})")


def main():
    args = parse_args()
    args.model_label = args.model_label or args.model
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
