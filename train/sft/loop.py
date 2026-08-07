"""Hand-written SFT training loop (no trl). Single-GPU and torchrun-DDP aware.

Uses AdamW(fused), cosine/warmup scheduling, bf16, gradient accumulation,
gradient checkpointing, and per-device batch size 1. Loss is computed inside
MemoryQwen3ForCausalLM (shifted CE over answer tokens, normalized per token).
"""
from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .lora import unwrap_model
from .param_groups import build_param_groups


@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    scheduler: str = "warmup_cosine"
    warmup_steps: int = 10
    constant_steps: int = 0
    max_steps: int = 200
    grad_accum: int = 1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 0
    save_every: int = 0
    output_dir: str = "outputs/sft/dev"
    grad_checkpointing: bool = False
    zero1: bool = False          # shard AdamW optimizer states across DDP ranks (ZeRO-1)
    seed: int = 0
    use_wandb: bool = False
    wandb_project: str = "memory-lm-sft"
    run_name: str = "dev"
    diag_side_norm: bool = True  # log side-branch output norm (gate/state health)


def build_optimizer(model: nn.Module, cfg: TrainConfig, is_ddp: bool = False) -> torch.optim.Optimizer:
    groups = build_param_groups(model, cfg.weight_decay)
    fused = torch.cuda.is_available()
    if cfg.zero1 and is_ddp:
        # ZeRO-1: shard optimizer states across ranks (~8GB/GPU at 2.33B trainable
        # -> 64k fits with headroom). Gradients stay full (DDP all-reduce), so
        # clip_grad_norm_ and no_sync are unaffected. Pairs with DDP's
        # gradient_as_bucket_view (set in train_sft). Per-group weight_decay is
        # carried in `groups`; ZeRO syncs lr from param_groups each step.
        from torch.distributed.optim import ZeroRedundancyOptimizer
        return ZeroRedundancyOptimizer(
            groups, optimizer_class=torch.optim.AdamW,
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=fused,
        )
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=fused)


def _scheduler_factor(cfg: TrainConfig, step: int) -> float:
    if cfg.scheduler not in {"warmup_cosine", "warmup_constant", "constant_cosine"}:
        raise ValueError(f"unknown scheduler {cfg.scheduler!r}")
    if cfg.scheduler in {"warmup_cosine", "warmup_constant"} and step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    if cfg.scheduler == "warmup_constant":
        return 1.0
    decay_start = cfg.warmup_steps if cfg.scheduler == "warmup_cosine" else cfg.constant_steps
    if cfg.scheduler == "constant_cosine" and step < decay_start:
        return 1.0
    prog = (step - decay_start) / max(1, cfg.max_steps - decay_start)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def build_scheduler(opt: torch.optim.Optimizer, cfg: TrainConfig):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _scheduler_factor(cfg, step))


def _ddp_info() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world, world > 1


class _SideNormProbe:
    """Records output-scale diagnostics at one memory layer.

    `side_out_abs` alone is not enough: a growing side branch can still be tiny
    relative to the total residual stream. The probed MemoryAttention computes
    detached scalar tensors only when `_record_o_stats` is set here, so normal
    layers do not pay the reduction cost.
    """

    def __init__(self, model: nn.Module):
        self.keys = (
            "main_out_abs",
            "side_out_abs",
            "total_out_abs",
            "side_out_ratio",
            "side_main_ratio",
        )
        self.stats = {k: 0.0 for k in self.keys}
        self.handle = None
        core = unwrap_model(model)
        for layer in core.model.layers:
            sa = layer.self_attn
            if hasattr(sa, "mem"):
                sa._record_o_stats = True
                self.handle = sa.register_forward_hook(self._hook)
                break

    def _hook(self, mod, inp, out):
        stats = getattr(mod, "_last_o_stats", {})
        for k in self.stats:
            if k in stats:
                self.stats[k] = float(stats[k].detach().cpu())

    @property
    def value(self) -> float:
        return self.stats["side_out_abs"]


def train(model, dataloader, cfg: TrainConfig, device, eval_fn=None):
    rank, local_rank, world, is_ddp = _ddp_info()
    is_main = rank == 0
    torch.manual_seed(cfg.seed + rank)

    if cfg.grad_checkpointing:
        core = model.module if hasattr(model, "module") else model
        core.gradient_checkpointing_enable()
        core.config.use_cache = False

    opt = build_optimizer(model, cfg, is_ddp)
    sched = build_scheduler(opt, cfg)
    probe = _SideNormProbe(model) if cfg.diag_side_norm else None

    wandb = None
    if cfg.use_wandb and is_main:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    model.train()
    step = 0
    data_iter = _cycle(dataloader)
    import time as _time
    _t_last = _time.time()
    _step_last = 0
    while step < cfg.max_steps:
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        loss_log_accum = 0.0
        for micro in range(cfg.grad_accum):
            # no_sync on all but the last micro-step: grad-accum then does ONE
            # DDP all-reduce per optimizer step, not grad_accum× redundant ones.
            is_last = micro == cfg.grad_accum - 1
            sync_ctx = (model.no_sync() if (is_ddp and not is_last
                        and hasattr(model, "no_sync")) else contextlib.nullcontext())
            with sync_ctx:
                batch = next(data_iter)
                batch = {k: v.to(device) for k, v in batch.items()}
                # Pass whatever per-token control / packing fields are present.
                model_kwargs = {
                    k: batch[k] for k in (
                        "is_evicted", "write_mask", "segment_ids",
                        "chunk_id", "evict_step",
                        "seq_ids", "cu_seqlens", "position_ids",
                    ) if k in batch
                }
                out = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    **model_kwargs,
                )
                micro_loss = out.loss
                loss = micro_loss / cfg.grad_accum
                loss.backward()
                loss_accum += loss.item()
                loss_log_accum += float(micro_loss.detach())

        gnorm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), cfg.grad_clip
        )
        opt.step()
        sched.step()
        step += 1

        should_log = step % cfg.log_every == 0 or step == 1
        loss_log = None
        if should_log:
            loss_log = torch.tensor(loss_log_accum, device=device, dtype=torch.float32)
            if is_ddp:
                torch.distributed.all_reduce(loss_log, op=torch.distributed.ReduceOp.SUM)
                loss_log /= world
            loss_log /= max(cfg.grad_accum, 1)
            if probe is not None:
                stat_tensor = torch.tensor(
                    [probe.stats[k] for k in probe.keys], device=device, dtype=torch.float32
                )
                if is_ddp:
                    torch.distributed.all_reduce(stat_tensor, op=torch.distributed.ReduceOp.SUM)
                    stat_tensor /= world
                for k, v in zip(probe.keys, stat_tensor.detach().cpu().tolist()):
                    probe.stats[k] = float(v)

        if is_main and should_log:
            lr = sched.get_last_lr()[0]
            now = _time.time()
            sps = (now - _t_last) / max(step - _step_last, 1)
            _t_last, _step_last = now, step
            eta_h = sps * (cfg.max_steps - step) / 3600
            msg = (f"step {step:>5}/{cfg.max_steps}  loss {float(loss_log):.4f}  lr {lr:.2e}  "
                   f"gnorm {gnorm:.2f}  {sps:.1f}s/step  eta {eta_h:.1f}h")
            if probe is not None:
                ps = probe.stats
                side_pct = ps["side_out_ratio"] * 100.0
                msg += (
                    f"  side|o| {ps['side_out_abs']:.3e}"
                    f"  side/total {side_pct:.2f}%"
                )
            print(msg, flush=True)
            if wandb is not None:
                log = {"loss": float(loss_log), "lr": lr, "grad_norm": float(gnorm)}
                if probe is not None:
                    log.update(probe.stats)
                    log["side_out_pct"] = probe.stats["side_out_ratio"] * 100.0
                wandb.log(log, step=step)

        if eval_fn is not None and cfg.eval_every and step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                metrics = eval_fn(model, step)
            if is_main and metrics:
                print(f"  [eval@{step}] {metrics}", flush=True)
                if wandb is not None:
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            model.train()

        if is_main and cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(model, cfg, step)

    if is_main:
        save_checkpoint(model, cfg, step)
        if wandb is not None:
            wandb.finish()
    return loss_accum


def save_checkpoint(model, cfg: TrainConfig, step: int) -> str:
    inner = model.module if hasattr(model, "module") else model
    out = os.path.join(cfg.output_dir, f"step_{step}")
    os.makedirs(out, exist_ok=True)
    # Save only TRAINABLE params (LoRA adapters and/or full-group base weights).
    # The frozen Qwen3 base is reconstructable via from_qwen3, so we don't dump
    # 12.7GB each time. Covers full / lora / mixed specs uniformly.
    trainable = {n: p.detach().cpu() for n, p in inner.named_parameters() if p.requires_grad}
    torch.save(trainable, os.path.join(out, "trainable.pt"))
    if hasattr(inner, "get_base_model"):  # also dump the PEFT adapter for easy reload
        inner.save_pretrained(out)
    print(f"  [ckpt] saved {out} ({len(trainable)} trainable tensors)", flush=True)
    return out


def _cycle(loader):
    epoch = 0
    while True:
        batch_sampler = getattr(loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        for b in loader:
            yield b
        epoch += 1
