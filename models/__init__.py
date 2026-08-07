"""Memory-augmented Qwen3 (RNN/GDN2 memory side-branch).

Import this package before `fla` so the vendored build of
flash-linear-attention (third_party/, which ships gdn2) shadows site-packages.
"""
from __future__ import annotations

from . import _bootstrap  # noqa: F401  (must run first)

from .configuration_memory_qwen3 import MemoryQwen3Config
from .memory_gdn2 import MemoryGatedDeltaNet2
from .modeling_memory_qwen3 import (
    MemoryAttention,
    MemoryQwen3ForCausalLM,
    MemoryQwen3Model,
    copy_init_side_branch,
    make_evict_and_mask,
    make_memory_and_mask,
)

__all__ = [
    "MemoryQwen3Config",
    "MemoryGatedDeltaNet2",
    "MemoryAttention",
    "MemoryQwen3Model",
    "MemoryQwen3ForCausalLM",
    "copy_init_side_branch",
    "make_evict_and_mask",
    "make_memory_and_mask",
]
