"""vLLM-side config for MemoryQwen3 = Qwen3 backbone + per-layer GDN2 side branch.

Reads the training checkpoint's fields (standard Qwen3 + the `mem_*` GDN2
hyper-parameters saved by `models/configuration_memory_qwen3.py`) and *additionally*
exposes the `linear_*` geometry that vLLM's mamba/GDN state machinery reads
(`MambaStateShapeCalculator.gated_delta_net_state_shape`).

Kept deliberately lightweight (only `PretrainedConfig`) so the plugin `register()`
can import it before CUDA init.
"""
from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class MemoryQwen3Config(PretrainedConfig):
    model_type = "memory_qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # ---- Qwen3 backbone geometry ----
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        rope_parameters: dict | None = None,
        rope_theta: float = 5_000_000.0,
        # ---- GDN2 side-branch (training `mem_*` names) ----
        mem_head_dim: int = 128,
        mem_num_heads: int = 32,
        mem_num_v_heads: int = 32,
        mem_expand_v: float = 1.0,
        mem_conv_size: int = 4,
        mem_conv_bias: bool = False,
        mem_norm_eps: float | None = None,
        memory_design: str = "X",
        layer_types: list[str] | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias

        # vLLM Qwen3Attention reads config.rope_parameters (a dict).
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_parameters.get("rope_theta", rope_theta)

        # GDN2 geometry: keep mem_* AND expose linear_* (what vLLM mamba reads).
        self.mem_head_dim = mem_head_dim
        self.mem_num_heads = mem_num_heads
        self.mem_num_v_heads = mem_num_v_heads
        self.mem_expand_v = mem_expand_v
        self.mem_conv_size = mem_conv_size
        self.mem_conv_bias = mem_conv_bias
        self.mem_norm_eps = mem_norm_eps if mem_norm_eps is not None else rms_norm_eps
        self.memory_design = memory_design

        self.linear_num_key_heads = mem_num_heads
        self.linear_num_value_heads = mem_num_v_heads
        self.linear_key_head_dim = mem_head_dim
        self.linear_value_head_dim = int(mem_head_dim * mem_expand_v)
        self.linear_conv_kernel_dim = mem_conv_size

        # Pass-through; only used by ModelConfig.is_hybrid gate (any non-"attention"
        # entry, or None, marks the model hybrid). We build attn+GDN2 on every layer.
        self.layer_types = layer_types

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
