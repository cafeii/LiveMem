"""SFT entrypoint (hand-written, no trl).

Single GPU:   PYTHONPATH=. python train/sft/train_sft.py --config train/sft/config/base.yaml
Multi-GPU:    PYTHONPATH=. torchrun --nproc_per_node=8 train/sft/train_sft.py --config ...
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

import models  # noqa: F401  (bootstraps vendored fla before any fla import)
from models import MemoryQwen3ForCausalLM
from train.sft.collator import collate_memory_packed, collate_memory_sft
from train.sft.data import (
    JsonlSFTDataset,
    MixedArrowDataset,
    SyntheticMemoryConfig,
    SyntheticMemoryDataset,
    SyntheticPackConfig,
    SyntheticPackDataset,
)
from train.sft.lora import apply_lora
from train.sft.loop import TrainConfig, train
from train.sft.param_groups import apply_trainable_spec, set_trainable_tier


def _deep_update(d: dict, key: str, val: str) -> None:
    cur = d
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    # best-effort type cast
    for cast in (int, float):
        try:
            val = cast(val); break
        except ValueError:
            pass
    if val in ("true", "false"):
        val = val == "true"
    cur[parts[-1]] = val


def build_model(mcfg: dict, device):
    mem_override_keys = (
        "mem_head_dim", "mem_num_heads", "mem_num_v_heads", "mem_expand_v",
        "mem_conv_size", "mem_conv_bias", "mem_norm_eps",
    )
    mem_overrides = {k: mcfg[k] for k in mem_override_keys if k in mcfg}
    init_merged = mcfg.get("init_from_merged")
    if init_merged:
        # A merged full MemoryQwen3 checkpoint already contains the initialized
        # side branch. Load it directly and let the caller apply the requested
        # trainable specification. This bypasses from_qwen3
        # copy-init (would re-zero the opened branch) and resume_trainable_from
        # (name/PEFT-prefix mismatch across a form change).
        model = MemoryQwen3ForCausalLM.from_pretrained(
            init_merged, dtype=torch.bfloat16,
            attn_implementation=mcfg.get("attn_implementation", "flex_attention"),
        )
        # fla keeps GDN A_log/dt_bias in fp32 (from_pretrained honors its fp32-keep);
        # force uniform bf16 like the from_qwen3 path so ZeRO-1's single-dense-dtype
        # requirement holds when the side branch is trained full-param.
        return model.to(device, torch.bfloat16)
    if mcfg.get("debug_small"):
        from models import MemoryQwen3Config
        cfg = MemoryQwen3Config(
            vocab_size=mcfg.get("vocab_size", 512), hidden_size=256, intermediate_size=512,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            max_position_embeddings=4096, memory_design=mcfg["memory_design"],
            mem_o_proj_zero_init=mcfg.get("mem_o_proj_zero_init", False),
            attn_implementation=mcfg.get("attn_implementation", "eager"),
            **mem_overrides,
        )
        model = MemoryQwen3ForCausalLM(cfg).to(device, torch.bfloat16)
    else:
        model = MemoryQwen3ForCausalLM.from_qwen3(
            mcfg["qwen3_path"], memory_design=mcfg["memory_design"],
            mem_layers=mcfg.get("mem_layers"),
            mem_o_proj_zero_init=mcfg.get("mem_o_proj_zero_init", True),
            dtype=torch.bfloat16,
            attn_implementation=mcfg.get("attn_implementation", "flex_attention"),
            **mem_overrides,
        ).to(device)
    return model


def load_trainable_checkpoint(model, path: str, rank: int = 0) -> None:
    """Initialize trainable weights from a save_checkpoint() trainable.pt file.

    Stage-to-stage training intentionally does not restore optimizer/scheduler
    state; each stage owns its LR schedule. Shape mismatches still raise via
    load_state_dict, while the large frozen backbone appears in `missing`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume_trainable_from not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing_trainable = sorted(trainable.difference(state))
    if missing_trainable:
        preview = ", ".join(missing_trainable[:8])
        raise RuntimeError(
            f"resume checkpoint is missing {len(missing_trainable)} trainable tensors: {preview}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(
            f"[resume] loaded trainable checkpoint {path} "
            f"({len(state)} tensors, missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )
        if unexpected:
            print(f"[resume] unexpected keys: {unexpected[:8]}", flush=True)


def _get_mix():
    # mix.py lives in tools/data_process (not a package); import by path.
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "dp_mix", os.path.join(repo, "tools", "data_process", "mix.py"))
    mix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mix)
    return mix


def build_mixture(items, seed, max_len=None, min_len=None):
    """Per-dataset: load -> length-filter -> fixed-seed sample -> concat+shuffle.
    The length filter runs before sampling so each
    requested count is drawn from the valid-length pool. Reuses mix's loader."""
    import random as _random
    import pyarrow as pa
    import pyarrow.compute as pc

    mix = _get_mix()
    tables = []
    for ds, sf, cnt in items:
        path = os.path.join(mix.TRAIN_DIR, ds, f"{sf}.arrow")
        if not os.path.exists(path):
            raise FileNotFoundError(f"训练数据不存在: {path}")
        # memory-map (shared OS page cache across DDP ranks -> NAS read once, not 8x)
        t = pa.ipc.open_file(pa.memory_map(path, "r")).read_all()
        if max_len is not None or min_len is not None:
            lens = pc.list_value_length(t.column("input_ids"))
            mask = None
            if max_len is not None:
                m = pc.less_equal(lens, max_len)
                mask = m if mask is None else pc.and_(mask, m)
            if min_len is not None:
                m = pc.greater_equal(lens, min_len)
                mask = m if mask is None else pc.and_(mask, m)
            n0 = t.num_rows
            t = t.filter(mask)
            print(f"[data] {ds}/{sf}: length-filter {n0} -> {t.num_rows} "
                  f"(max_len={max_len}, min_len={min_len})")
        t = mix.sample_table(t, cnt, seed)
        tables.append(t)
    combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    # No global shuffle/take here: take() over the full corpus overflows int32
    # list offsets (5.9B tokens > 2.1B). Row order is irrelevant downstream —
    # PackPlanner shuffles at the pack level (seed) and _bin_pack sorts by length.
    return combined


def _system_segment_ids(qwen3_path):
    """Tokenize format.py's system_segment() once (the sink prompt)."""
    import importlib.util
    from transformers import AutoTokenizer
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = importlib.util.spec_from_file_location(
        "dp_format", os.path.join(repo, "tools", "data_process", "format.py"))
    fmt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fmt)
    tok = AutoTokenizer.from_pretrained(qwen3_path)
    return tok(fmt.system_segment(), add_special_tokens=False).input_ids


def build_dataset(dcfg: dict, qwen3_path: str | None = None):
    fields = {k: v for k, v in dcfg.items() if k not in ("kind", "pack")}
    if dcfg["kind"] == "synthetic":
        return SyntheticMemoryDataset(SyntheticMemoryConfig(**fields))
    if dcfg["kind"] == "synthetic_pack":
        return SyntheticPackDataset(SyntheticPackConfig(**fields))
    if dcfg["kind"] == "jsonl":
        return JsonlSFTDataset(dcfg["path"])
    if dcfg["kind"] == "mixture":
        # data.datasets: [[name, split, count|null], ...]  (count null = all)
        # optional data.max_len / data.min_len: training-side token-length filter.
        items = [(d[0], d[1], d[2] if len(d) > 2 else None) for d in dcfg["datasets"]]
        # inject_system: prepend format.py SYSTEM_PROMPT as a never-evicted sink
        # chunk 0. n_sink>=1 follows automatically in the dataset.
        # Tokenize it FIRST so the length filter reserves its budget — every
        # example + sink must stay <= max_len (= pack target), else the padded
        # collator overflows the target.
        system_ids = _system_segment_ids(qwen3_path) if dcfg.get("inject_system") else None
        sys_len = len(system_ids) if system_ids else 0
        max_len = dcfg.get("max_len")
        table = build_mixture(items, seed=dcfg.get("seed", 0),
                              max_len=(max_len - sys_len) if max_len else None,
                              min_len=dcfg.get("min_len"))
        return MixedArrowDataset(table, n_sink=dcfg.get("n_sink", 0), system_ids=system_ids,
                                 w0_prob=dcfg.get("w0_prob", 0.0), w_seed=dcfg.get("seed", 0))
    raise ValueError(f"unknown data.kind {dcfg['kind']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.set:
        k, v = kv.split("=", 1)
        _deep_update(cfg, k, v)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        from datetime import timedelta
        # 30min watchdog: a slow rank (e.g. first-shape flex compile) must not
        # trip the default 10min NCCL timeout while others wait in all-reduce.
        torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)

    model = build_model(cfg["model"], device)
    tcfg = cfg["train"]
    lora_cfg = cfg.get("lora") or tcfg.get("lora") or {}
    if "trainable" in tcfg:
        # Configure each parameter group independently as full, LoRA, or frozen.
        model, summary = apply_trainable_spec(
            model, tcfg["trainable"], lora_cfg=lora_cfg,
            train_norms=tcfg.get("train_norms", False), train_embed=tcfg.get("train_embed", False))
        if local_rank == 0:
            print(f"[params] {summary['spec']} | full={summary['full_groups']} lora={summary['lora_groups']} "
                  f"| trainable={summary['n_trainable']/1e6:.1f}M / {summary['n_total']/1e6:.1f}M "
                  f"({summary['pct']:.2f}%)")
    else:  # Compatibility path: train.tier with global lora.enable.
        tier = tcfg.get("tier", "L1")
        if lora_cfg.get("enable"):
            model = apply_lora(model, tier=tier, r=lora_cfg.get("r", 16),
                               alpha=lora_cfg.get("alpha", 32), dropout=lora_cfg.get("dropout", 0.0))
        else:
            summary = set_trainable_tier(model, tier, train_norms=tcfg.get("train_norms", False),
                                         train_embed=tcfg.get("train_embed", False))
            if local_rank == 0:
                print(f"[params] tier={tier} trainable={summary['n_trainable']/1e6:.1f}M "
                      f"({summary['pct']:.2f}%)")

    global_rank = int(os.environ.get("RANK", 0))
    resume_path = tcfg.get("resume_trainable_from")
    if resume_path:
        load_trainable_checkpoint(model, resume_path, rank=global_rank)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
            gradient_as_bucket_view=True,  # .grad aliases the reduce bucket (saves ~1× grad)
        )

    ds = build_dataset(cfg["data"], qwen3_path=cfg["model"].get("qwen3_path"))
    rank = int(os.environ.get("RANK", 0))
    if cfg["data"].get("pack"):
        # Online deterministic packing to cfg.data.max_len (= cu_seqlens target):
        # A bounded padded shape prevents repeated flex compilation and limits
        # memory use. Per-device micro-batch = one [1, target] pack; batch scale
        # = world*grad_accum, with label-balanced steps (PackPlanner).
        from functools import partial
        from train.sft.data import PackPlanner
        target = cfg["data"].get("max_len", 32768)
        ga = cfg["train"].get("loop", {}).get("grad_accum", 1)
        pack_bs = cfg["train"].get("batch_size", 1)
        planner = PackPlanner(ds.lengths, ds.label_counts, target=target, world=world,
                              rank=rank, grad_accum=ga, micro_batch_size=pack_bs,
                              seed=cfg["data"].get("seed", 0))
        core = model.module if hasattr(model, "module") else model
        eos = getattr(core.config, "eos_token_id", 0)
        pad_id = (eos[0] if isinstance(eos, (list, tuple)) else eos) or 0
        collate = partial(collate_memory_packed, target=target, pad_id=pad_id)
        if rank == 0:
            print(f"[data] online packing to {target} tok | pack_bs={pack_bs} | grad_accum={ga} | "
                  f"global_packs/step={world * ga * pack_bs} | "
                  f"{len(ds)} samples -> {len(planner)} micro_batches/rank "
                  f"({len(planner) // max(ga, 1)} optimizer steps/epoch)")
        loader = DataLoader(ds, batch_sampler=planner, collate_fn=collate)
    else:
        sampler = None
        if world > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                         seed=cfg["data"].get("seed", 0), drop_last=True)
        loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 1),
                            shuffle=(sampler is None), sampler=sampler,
                            collate_fn=collate_memory_sft, drop_last=False)

    tc = TrainConfig(**cfg["train"].get("loop", {}))
    train(model, loader, tc, device)


if __name__ == "__main__":
    main()
