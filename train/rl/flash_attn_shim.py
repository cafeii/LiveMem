"""Pure-torch shim for ``flash_attn.bert_padding``.

verl's trainer-side layout conversion (workers/utils/padding.py) hard-imports
`flash_attn.bert_padding.{unpad_input,pad_input,index_first_axis,rearrange}` —
pure tensor utilities, no CUDA kernels. Our venv deliberately has no flash-attn
build (the memory model trains on flex attention). transformers ships identical
implementations, so when the real package is absent we register a module shim
in sys.modules instead of compiling flash-attn.

Safe by construction: `importlib.metadata.version("flash_attn")` still raises
PackageNotFoundError, so transformers' `is_flash_attn_2_available()` stays
False — no attention-implementation selection is affected. Imported from
train/rl/memory_hf_registry.py (verl external_lib), i.e. only in trainer
processes that need it.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types


def install() -> None:
    try:
        import flash_attn.bert_padding  # noqa: F401  (real package wins)
        return
    except ImportError:
        pass
    if "flash_attn.bert_padding" in sys.modules:
        return

    from einops import rearrange
    from transformers.modeling_flash_attention_utils import (
        _index_first_axis,
        _pad_input,
        _unpad_input,
    )

    pkg = types.ModuleType("flash_attn")
    pkg.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None, is_package=True)
    pkg.__path__ = []
    sub = types.ModuleType("flash_attn.bert_padding")
    sub.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    sub.unpad_input = _unpad_input
    sub.pad_input = _pad_input
    sub.index_first_axis = _index_first_axis
    sub.rearrange = rearrange
    pkg.bert_padding = sub
    sys.modules["flash_attn"] = pkg
    sys.modules["flash_attn.bert_padding"] = sub
