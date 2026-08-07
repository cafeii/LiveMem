"""MemoryQwen3 vLLM integration (non-invasive external package).

Lives under the workspace `vllm/` folder per project convention; the importable
package name is `memory_qwen3` (NOT `vllm`) to avoid shadowing the real engine.
"""
from __future__ import annotations

from .configuration_memory_qwen3 import MemoryQwen3Config

__all__ = ["MemoryQwen3Config", "register"]


def register() -> None:
    from .registry import register as _register

    _register()
