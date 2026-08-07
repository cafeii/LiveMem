"""Trainable-parameter selection and AdamW parameter groups.

Two orthogonal axes define the trainable specification:
  position (module group):  rnn (`.self_attn.mem.`) / attn (`.self_attn.attn.`) / ffn (`.mlp.`)
  form per group:           full | lora | frozen
`apply_trainable_spec` sets each group independently, e.g. rnn=full, attn=lora,
ffn=frozen. Embedding / final norm / lm_head stay frozen unless explicitly enabled.

`TIER_PATTERNS` retains the cumulative L1/L2/L3 compatibility interface.
"""
from __future__ import annotations

import torch.nn as nn

_SIDE = ".self_attn.mem."
_ATTN = ".self_attn.attn."
_FFN = ".mlp."

GROUP_PATTERNS = {"rnn": _SIDE, "attn": _ATTN, "ffn": _FFN}
VALID_FORMS = ("full", "lora", "frozen")

TIER_PATTERNS = {
    "L1": (_SIDE,),
    "L2": (_SIDE, _ATTN),
    "L3": (_SIDE, _ATTN, _FFN),
}


def apply_trainable_spec(model, spec: dict, lora_cfg: dict | None = None,
                         train_norms: bool = False, train_embed: bool = False):
    """Set each module group's form. `spec` e.g. {"rnn":"full","attn":"lora","ffn":"frozen"}.
    Returns (model_or_peft, summary). full+lora may coexist: LoRA-wrap the lora
    groups (PEFT freezes the base + adds adapters), then re-enable requires_grad on
    the full groups' base params. Returns a PEFT-wrapped model iff any group=lora."""
    for g, v in spec.items():
        if g not in GROUP_PATTERNS or v not in VALID_FORMS:
            raise ValueError(f"bad trainable spec entry {g}={v}; groups={list(GROUP_PATTERNS)} forms={VALID_FORMS}")
    lora_groups = [g for g, v in spec.items() if v == "lora"]
    full_groups = [g for g, v in spec.items() if v == "full"]
    if not lora_groups and not full_groups:
        raise ValueError("trainable spec trains nothing (all frozen)")

    if lora_groups:
        from peft import LoraConfig, get_peft_model
        targets = [
            n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and any(GROUP_PATTERNS[g] in n for g in lora_groups)
        ]
        lc = lora_cfg or {}
        # autocast_adapter_dtype=False keeps LoRA adapters in the base dtype (bf16)
        # instead of PEFT's default fp32 — so ALL trainables share one dtype and
        # ZeroRedundancyOptimizer (ZeRO-1) accepts them (it rejects mixed dtypes).
        model = get_peft_model(model, LoraConfig(
            r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32), lora_dropout=lc.get("dropout", 0.0),
            target_modules=targets, bias="none", task_type="CAUSAL_LM"),
            autocast_adapter_dtype=False)
        base = model.get_base_model()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        base = model

    for n, p in base.named_parameters():
        if any(GROUP_PATTERNS[g] in n for g in full_groups):
            p.requires_grad_(True)
        # LoRA groups: PEFT only adapts nn.Linear. Also FULL-train this group's
        # non-Linear params (RNN A_log/dt_bias/short-conv/o_norm; attn q/k_norm) —
        # they're new/from-scratch and would never learn if left frozen at init.
        # (`base_layer`/`lora_` are the LoRA-wrapped Linear's own tensors; skip.)
        if (any(GROUP_PATTERNS[g] in n for g in lora_groups)
                and "base_layer" not in n and "lora_" not in n):
            p.requires_grad_(True)
        if train_norms and ("norm" in n.lower()):
            p.requires_grad_(True)
        if train_embed and ("embed_tokens" in n or "lm_head" in n):
            p.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    summary = {"spec": dict(spec), "lora_groups": lora_groups, "full_groups": full_groups,
               "n_trainable": n_train, "n_total": n_total, "pct": 100.0 * n_train / max(n_total, 1)}
    return model, summary


def set_trainable_tier(
    model: nn.Module,
    tier: str,
    train_norms: bool = False,
    train_embed: bool = False,
) -> dict:
    """Freeze everything, then unfreeze the params matching `tier`.
    Returns a small summary {tier, n_trainable, n_total}."""
    if tier not in TIER_PATTERNS:
        raise ValueError(f"tier must be one of {list(TIER_PATTERNS)}, got {tier!r}")
    pats = TIER_PATTERNS[tier]

    n_train = n_total = 0
    for name, p in model.named_parameters():
        trainable = any(pat in name for pat in pats)
        if train_norms and (".norm" in name or "layernorm" in name.lower()):
            trainable = True
        if train_embed and ("embed_tokens" in name or "lm_head" in name):
            trainable = True
        p.requires_grad_(trainable)
        n_total += p.numel()
        if trainable:
            n_train += p.numel()
    return {"tier": tier, "n_trainable": n_train, "n_total": n_total,
            "pct": 100.0 * n_train / max(n_total, 1)}


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split trainable params into decay / no-decay groups.
    No decay for 1-D tensors (norms/biases) and fla params flagged
    `_no_weight_decay` (A_log, dt_bias)."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or getattr(p, "_no_weight_decay", False):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
