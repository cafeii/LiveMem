"""Non-invasive registration of MemoryQwen3 into vLLM.

Exposed as a `vllm.general_plugins` entry point (see pyproject.toml) so vLLM
runs `register()` in every process (engine core / workers) at startup. Also
callable directly from a launcher before `LLM(...)`.

Must be idempotent (re-run per process) and must NOT import CUDA-touching
modules at top level — the model class is registered lazily by string.
"""
from __future__ import annotations

_DONE = False


def register() -> None:
    global _DONE
    if _DONE:
        return

    from vllm import ModelRegistry

    # Lazy string form: model module is imported only inside worker processes,
    # avoiding CUDA re-init in a forked parent.
    ModelRegistry.register_model(
        "MemoryQwen3ForCausalLM",
        "memory_qwen3.modeling_memory_qwen3:MemoryQwen3ForCausalLM",
    )

    # Teach HF AutoConfig about model_type="memory_qwen3" so checkpoints load.
    from transformers import AutoConfig

    from memory_qwen3.configuration_memory_qwen3 import MemoryQwen3Config

    AutoConfig.register("memory_qwen3", MemoryQwen3Config, exist_ok=True)

    # Per-request eviction hooks are idempotent and no-ops
    # unless env MEM_SERVE_EVICT=1, and each only patches the class present in its
    # own process (engine: Scheduler; worker: GPUModelRunner). The worker hook
    # also imports flashinfer_evict (registers EvictFlashInferBackend) — kept out
    # of the engine path to avoid touching the attention backend in the forked
    # engine-core parent.
    from memory_qwen3.evict_registry import (
        install_engine_hooks,
        install_worker_hooks,
    )

    install_engine_hooks()
    install_worker_hooks()

    _DONE = True
