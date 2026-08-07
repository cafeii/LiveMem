"""Config for the memory-augmented Qwen3 (main Qwen3Attention path ‖ GDN2 side path)."""
from __future__ import annotations

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class MemoryQwen3Config(Qwen3Config):
    """Qwen3 + per-attention-head GDN2 memory side-branch.

    All base Qwen3 fields are inherited unchanged. The `mem_*` fields configure
    the side branch and the two memory mechanisms (Design X / Design Y).
    """

    model_type = "memory_qwen3"

    def __init__(
        self,
        # Which mechanism: "X" = continuous scan (write_mask=None);
        #                  "Y" = gated read/write separation (freeze gates on read tokens).
        memory_design: str = "Y",
        # layers that get a memory branch; None = all layers.
        mem_layers: list[int] | None = None,
        # GDN2 side-branch hyperparameters. The default geometry is a full-MHA
        # copy of Qwen3 (8 KV heads repeated 4x -> 32 heads). Configurations can
        # explicitly set mem_num_heads=8,
        # mem_num_v_heads=8, mem_expand_v=4 to keep similar state capacity with
        # fewer Q/K heads.
        mem_head_dim: int | None = None,
        mem_num_heads: int | None = None,
        mem_num_v_heads: int | None = None,
        mem_expand_v: float = 1.0,
        mem_conv_size: int = 4,
        mem_conv_bias: bool = False,
        mem_norm_eps: float | None = None,
        # Zero-initialize the side o_proj so training starts near the original Qwen3.
        mem_o_proj_zero_init: bool = True,
        # In RL mode, derive chunk_id/evict_step inside forward when they are not
        # passed explicitly. The derivation uses the serve-shared stride
        # policy (train/sft/eviction_policy.py) — training mask ≡ vllm rollout.
        mem_stride_policy: bool = False,
        # RL trainable specification: full-parameter side branch, frozen main path.
        # SFT uses train/sft trainable specs instead; this flag is for trainers
        # (verl) that cannot call apply_trainable_spec.
        mem_freeze_main_path: bool = False,
        # For fixed-shape RL packed rows, right-pad a
        # [1, T] full forward up to the next multiple of this value so flex
        # compiles a bounded set of shapes (65536 -> one shape for <=64k packs;
        # 0 = off). Pad span is an isolated tail segment, sliced off before
        # returning hidden states.
        mem_rl_pad_multiple: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if memory_design not in ("X", "Y"):
            raise ValueError(f"memory_design must be 'X' or 'Y', got {memory_design!r}")
        self.memory_design = memory_design
        self.mem_layers = mem_layers
        self.mem_head_dim = mem_head_dim if mem_head_dim is not None else self.head_dim
        self.mem_num_heads = (
            mem_num_heads if mem_num_heads is not None else self.num_attention_heads
        )
        self.mem_num_v_heads = (
            mem_num_v_heads if mem_num_v_heads is not None else self.mem_num_heads
        )
        self.mem_expand_v = mem_expand_v
        self.mem_conv_size = mem_conv_size
        self.mem_conv_bias = mem_conv_bias
        self.mem_norm_eps = mem_norm_eps if mem_norm_eps is not None else self.rms_norm_eps
        self.mem_o_proj_zero_init = mem_o_proj_zero_init
        self.mem_stride_policy = mem_stride_policy
        self.mem_freeze_main_path = mem_freeze_main_path
        self.mem_rl_pad_multiple = mem_rl_pad_multiple

    @property
    def memory_layer_indices(self) -> list[int]:
        if self.mem_layers is None:
            return list(range(self.num_hidden_layers))
        return list(self.mem_layers)
