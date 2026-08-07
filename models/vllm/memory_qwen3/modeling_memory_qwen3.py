"""MemoryQwen3 for vLLM 0.19.1: Qwen3 backbone with a per-layer parallel GDN2
side branch in parallel. Each decoder layer runs Qwen3 attention (main, paged-KV)
‖ GDN2 (side, mamba state) and sums: o = o_main + o_side, then the usual MLP.

Structurally mirrors FalconH1ParallelHybrid (same-layer attn+mamba, all layers),
but the backbone/attn are Qwen3 and the side branch is our GDN2.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .gdn2_side import MemoryGDN2Attention


class MemoryQwen3ParallelHybrid(nn.Module):
    """One decoder layer: main Qwen3 attention path ‖ GDN2 side path, o = o_main + o_side."""

    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        cache_config,
        quant_config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # Main path: vLLM Qwen3 attention (paged-KV). prefix carries the layer idx.
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=AttentionType.DECODER,
        )

        # ``CHUNK_EVICT=sink_len,keep_recent`` enables physical KV eviction.
        # It makes the main-path attention emit ChunkEvictSpec so evicted middle
        # blocks are freed at runtime (peak KV bounded). Non-invasive (instance
        # patch of get_kv_cache_spec); disabled by default.
        _ce = os.environ.get("CHUNK_EVICT")
        if _ce:
            from .evict_manager import patch_attention_for_evict
            _sink, _keep = (int(x) for x in _ce.split(","))
            patch_attention_for_evict(self.self_attn.attn, _sink, _keep)

        # Side path: GDN2. Use a *distinct* forward-context prefix
        # (num_hidden_layers + layer_idx) so its state spec key never collides
        # with the main attention layer's (mirrors FalconH1's ssm_layer_idx).
        parent = prefix.rsplit(".", 1)[0]  # "....model.layers"
        gdn_idx = config.num_hidden_layers + layer_idx
        self.mem = MemoryGDN2Attention(
            config, vllm_config=vllm_config, prefix=f"{parent}.{gdn_idx}.mem"
        )

        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Combine the main and side paths elementwise.
        attn_out = self.self_attn(positions, hidden_states)
        # zeros (not empty): the GDN custom op's fake impl doesn't write padding
        # rows, so under cudagraph capture empty would leave garbage there that
        # propagates through the add/MLP (matches native, vllm PR #28182).
        side_out = torch.zeros_like(hidden_states)
        self.mem(hidden_states, side_out)
        hidden_states = attn_out + side_out

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class MemoryQwen3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size)
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )

        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            return MemoryQwen3ParallelHybrid(
                config,
                layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                vllm_config=vllm_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MemoryQwen3ForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # --- hybrid (mamba) state contract: GDN2 across ALL layers --------------
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        conv_dtype, _ = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        # fla GDN2 kernels require fp32 recurrent state (must match the
        # layer's get_state_dtype so page-size alignment is consistent).
        return conv_dtype, torch.float32

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        pc = vllm_config.parallel_config
        hf = vllm_config.model_config.hf_config
        spec = vllm_config.speculative_config
        num_spec = spec.num_speculative_tokens if spec else 0
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            pc.tensor_parallel_size,
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            hf.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    # ----------------------------------------------------------------------
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = MemoryQwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map the training MemoryQwen3 checkpoint onto the vLLM module tree:
          - main attn `...self_attn.attn.{q,k,v}_proj` -> stacked `self_attn.qkv_proj`
            (training wraps base Qwen3Attention as `self_attn.attn.*`; unwrap it).
          - main attn `...self_attn.attn.{o_proj,q_norm,k_norm}` -> `self_attn.*`.
          - GDN2 side `...self_attn.mem.*` -> `mem.mem.*` (reused MemoryGatedDeltaNet2),
            loaded by name (separate q/k/v projections — no stacking).
          - mlp `{gate,up}_proj` -> stacked `gate_up_proj`; embed/norm/lm_head direct.
        """
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            if ".self_attn.mem." in name:
                # GDN2 side branch -> the reused MemoryGatedDeltaNet2 at `mem.mem.*`.
                name = name.replace(".self_attn.mem.", ".mem.mem.")
                if name not in params:
                    continue
                p = params[name]
                getattr(p, "weight_loader", default_weight_loader)(p, w)
                loaded.add(name)
                continue
            name = name.replace(".self_attn.attn.", ".self_attn.")  # unwrap main attn
            if name.startswith("lm_head") and self.config.tie_word_embeddings:
                continue
            done = False
            for tgt_sub, src_sub, shard_id in stacked:
                if src_sub in name:
                    tgt = name.replace(src_sub, tgt_sub)
                    if tgt not in params:
                        continue
                    p = params[tgt]
                    p.weight_loader(p, w, shard_id)
                    loaded.add(tgt)
                    done = True
                    break
            if done:
                continue
            if name not in params:
                continue
            p = params[name]
            getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(name)
        return loaded
