"""GDN2 memory side-branch implemented as a vLLM MambaBase layer.

Reuses the *training* module `models.memory_gdn2.MemoryGatedDeltaNet2` for all
submodules (q/k/v/b/w/f/g_proj, A_log, dt_bias, q/k/v short-conv, o_norm,
o_proj) — so the training checkpoint loads as-is AND the math is bit-for-bit the
same as training. Only the state plumbing is swapped to vLLM's paged slots:
  - conv: vLLM causal_conv1d_fn (prefill) / causal_conv1d_update (decode) over
    the paged conv_state, with q/k/v conv weights concatenated into one depthwise
    kernel (depthwise conv over [q;k;v] == three separate convs).
  - ssm: fla chunk_gdn2 (prefill) / fused_recurrent_gdn2 (decode), gathering
    the initial recurrent state from ssm_state[indices] and scattering the final
    state back (truncate-mode / single-point reuse done outside the kernel).

This implementation uses continuous scanning without ``write_mask``.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

# Repo root on path so we can reuse the training GDN2 module (this file lives
# at <root>/models/vllm/memory_qwen3/). Appending (not prepending) keeps the
# real `vllm` package (site-packages) winning import resolution.
_WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _WS not in sys.path:
    sys.path.append(_WS)

from models.memory_gdn2 import MemoryGatedDeltaNet2  # runs _bootstrap (vendored fla)
from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2  # noqa: E402

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import direct_register_custom_op


_GDN_PREFILL_TOKENS = {"n": 0}


def gdn2_paged_compute(
    mem: MemoryGatedDeltaNet2,
    conv_weight: torch.Tensor,  # [conv_dim, K] depthwise (q|k|v concatenated)
    hidden: torch.Tensor,  # [num_actual, hidden]
    conv_state: torch.Tensor,  # [n_blk, conv_dim, K-1]  (already transposed)
    ssm_state: torch.Tensor,  # [n_blk, HV, K, V]
    state_idx: torch.Tensor,  # [n_seq] slot per sequence
    query_start_loc: torch.Tensor,  # cu_seqlens [n_seq+1]
    has_initial_state: torch.Tensor | None,  # [n_seq] bool, or None (decode)
    is_prefill: bool,
) -> torch.Tensor:
    """Core GDN2 forward over vLLM paged state. Returns o [num_actual, hidden].
    Self-contained (no forward_context) so it can be unit-tested standalone."""
    n = hidden.shape[0]
    if is_prefill:
        _GDN_PREFILL_TOKENS["n"] += int(n)
    Hk, Hv = mem.num_heads, mem.num_v_heads
    Dk, Dv = mem.head_k_dim, mem.head_v_dim

    # --- input projections + short conv (depthwise, silu), via paged conv_state
    mixed = torch.cat([mem.q_proj(hidden), mem.k_proj(hidden), mem.v_proj(hidden)], dim=-1)
    if is_prefill:
        mixed = causal_conv1d_fn(
            mixed.transpose(0, 1).contiguous(),
            conv_weight,
            bias=None,
            conv_states=conv_state,
            query_start_loc=query_start_loc,
            cache_indices=state_idx,
            has_initial_state=has_initial_state,
            activation="silu",
        ).transpose(0, 1)
    else:
        mixed = causal_conv1d_update(
            mixed,
            conv_state,
            conv_weight,
            bias=None,
            activation="silu",
            conv_state_indices=state_idx[:n],
        )
    q, k, v = mixed.split([mem.key_dim, mem.key_dim, mem.value_dim], dim=-1)

    # --- gates (computed outside the kernel, exactly as training) ----------
    g = F.softplus(mem.f_proj(hidden).float() + mem.dt_bias)
    b = mem.b_proj(hidden).sigmoid()
    w = mem.w_proj(hidden).sigmoid()

    q = q.view(1, n, Hk, Dk)
    k = k.view(1, n, Hk, Dk)
    g = g.view(1, n, Hk, Dk)
    b = b.view(1, n, Hk, Dk)
    v = v.view(1, n, Hv, Dv)
    w = w.view(1, n, Hv, Dv)
    g = -mem.A_log.float().exp().unsqueeze(-1) * g
    if Hv > Hk:
        from einops import repeat

        q, k, g, b = (repeat(x, "... h d -> ... (h g) d", g=Hv // Hk) for x in (q, k, g, b))
    if mem.allow_neg_eigval:
        b = b * 2.0

    # --- recurrent state: gather initial -> kernel -> scatter final --------
    init = ssm_state[state_idx].contiguous()
    kernel = chunk_gdn2 if is_prefill else fused_recurrent_gdn2
    if is_prefill and has_initial_state is not None:
        init = init.clone()
        init[~has_initial_state] = 0
    o, final = kernel(
        q=q, k=k, v=v, g=g, b=b, w=w,
        initial_state=init,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
    )
    ssm_state[state_idx] = final.to(ssm_state.dtype)

    # --- gated RMSNorm + output projection ---------------------------------
    o = mem.o_norm(o, mem.g_proj(hidden).view(1, n, Hv, Dv))
    o = o.reshape(n, Hv * Dv)
    return mem.o_proj(o)


# --- cuda-graph: GDN core as an opaque vLLM custom op ----------------------
# Wraps gdn2_paged_compute so VLLM_COMPILE treats it as a black box (not traced
# into the fx graph). Combined with adding "vllm::memory_gdn2_core" to
# compilation_config.splitting_ops (in __init__), the GDN side runs eager OUTSIDE
# the piecewise cudagraph — exactly like native mamba/GDN (gdn_linear_attn.py).
# Its in-place paged-state writes (conv/ssm) + triton kernels + dynamic indexing
# are all cudagraph-hostile and must stay out of the captured graph; the paged
# state is reached via self.kv_cache inside the op (NOT a traced arg), so it is
# invisible to the compiler. Only `output` is a mutated traced arg.
def _memory_gdn2_core(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:  # profile / warmup run
        return
    self = ctx.no_compile_layers[layer_name]
    m = attn_metadata[layer_name]
    n = m.num_actual_tokens
    is_prefill = m.num_prefills > 0
    o = gdn2_paged_compute(
        self.mem,
        self._conv_weight(),
        hidden_states[:n],
        self.kv_cache[0].transpose(-1, -2),
        self.kv_cache[1],
        m.non_spec_state_indices_tensor,
        m.non_spec_query_start_loc,
        m.has_initial_state if is_prefill else None,
        is_prefill,
    )
    output[:n] = o


def _memory_gdn2_core_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="memory_gdn2_core",
    op_func=_memory_gdn2_core,
    mutates_args=["output"],
    fake_impl=_memory_gdn2_core_fake,
)


class MemoryGDN2Attention(nn.Module, MambaBase):
    """Qwen3 per-layer GDN2 side branch over vLLM paged state."""

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"  # reuse GDNAttentionBackend + GDNAttentionMetadata

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        conv_dtype, _ = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )
        # fla chunk_gdn2 / fused_recurrent_gdn2 require fp32 recurrent state.
        return conv_dtype, torch.float32

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # conv: (K-1+num_spec, conv_dim/tp); ssm: (HV/tp, head_k_dim, head_v_dim).
        # ssm laid out (K, V) to match chunk_gdn2's default (state_v_first=False),
        # so gather/scatter round-trips without transpose.
        conv_dim = (self.head_k_dim * self.num_k_heads * 2 + self.head_v_dim * self.num_v_heads)
        conv_state_shape = (self.conv_kernel_size - 1 + self.num_spec, conv_dim // self.tp_size)
        ssm_state_shape = (self.num_v_heads // self.tp_size, self.head_k_dim, self.head_v_dim)
        return conv_state_shape, ssm_state_shape

    def __init__(self, config, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        spec = vllm_config.speculative_config
        self.num_spec = spec.num_speculative_tokens if spec else 0

        # Reuse the training GDN2 module: identical submodules/weights/math.
        self.mem = MemoryGatedDeltaNet2(
            hidden_size=config.hidden_size,
            expand_v=config.mem_expand_v,
            head_dim=config.mem_head_dim,
            num_heads=config.mem_num_heads,
            num_v_heads=config.mem_num_v_heads,
            mode="chunk",
            use_short_conv=True,
            conv_size=config.mem_conv_size,
            conv_bias=config.mem_conv_bias,
            layer_idx=self.layer_idx,
            norm_eps=config.mem_norm_eps,
        )

        self.kv_cache = (torch.tensor([]), torch.tensor([]))
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # cuda graph: keep our GDN core op OUT of the piecewise cudagraph pieces.
        # Native ops are hardcoded in CompilationConfig._attention_ops; ours is
        # not, so inject it into splitting_ops (already finalized to a list by
        # set_splitting_ops_for_v1 before model construction). Idempotent.
        _op = "vllm::memory_gdn2_core"
        if (compilation_config.splitting_ops is not None
                and _op not in compilation_config.splitting_ops):
            compilation_config.splitting_ops.append(_op)

    def _conv_weight(self) -> torch.Tensor:
        return torch.cat(
            [self.mem.q_conv1d.weight, self.mem.k_conv1d.weight, self.mem.v_conv1d.weight],
            dim=0,
        ).squeeze(1)  # [conv_dim, K]

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        # Opaque custom op (see _memory_gdn2_core): runs eager outside the
        # piecewise cudagraph; profile/None handling lives inside the op.
        torch.ops.vllm.memory_gdn2_core(hidden_states, output, self.prefix)
