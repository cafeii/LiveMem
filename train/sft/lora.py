"""Optional LoRA applied to a selected parameter tier.

`apply_lora(model, tier=...)` puts LoRA adapters on exactly the nn.Linear leaves
of the chosen tier (L1 side branch / L2 +main attn / L3 +FFN), freezing the base.
"""
from __future__ import annotations

import torch.nn as nn

from .param_groups import TIER_PATTERNS


def linear_names_in_tier(model: nn.Module, tier: str) -> list[str]:
    if tier not in TIER_PATTERNS:
        raise ValueError(f"tier must be one of {list(TIER_PATTERNS)}, got {tier!r}")
    pats = TIER_PATTERNS[tier]
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(p in name for p in pats)
    ]


def unwrap_model(model: nn.Module) -> nn.Module:
    """Descend through DDP (.module) and PEFT (.get_base_model) wrappers to the
    underlying MemoryQwen3ForCausalLM (for probes / checkpoint helpers)."""
    m = model
    if hasattr(m, "module"):  # DDP
        m = m.module
    if hasattr(m, "get_base_model"):  # PEFT
        m = m.get_base_model()
    return m


def apply_lora(model, tier: str = "L1", r: int = 16, alpha: int = 32, dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model

    targets = linear_names_in_tier(model, tier)
    if not targets:
        raise ValueError(f"no Linear modules found for tier {tier!r}")
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)
