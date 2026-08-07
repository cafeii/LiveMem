"""Memory-augmented Qwen3: main Qwen3Attention path ‖ GDN2 side path, o = o_main + o_side.

Design X (continuous scan) and Design Y (gated read/write) share one skeleton
and the same attention eviction mask; they differ only in the RNN `write_mask`:
  - X: write_mask = None   (RNN scans every token; state = compression of all)
  - Y: write_mask = is_evicted (open gate only on the evict/compress region)
"""
from __future__ import annotations

from . import _bootstrap  # noqa: F401  (vendored fla on path before any fla import)

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)

from .configuration_memory_qwen3 import MemoryQwen3Config
from .memory_gdn2 import MemoryGatedDeltaNet2


def make_memory_and_mask(
    is_evicted: torch.Tensor | None = None,
    segment_ids: torch.Tensor | None = None,
    seq_ids: torch.Tensor | None = None,
    chunk_id: torch.Tensor | None = None,
    evict_step: torch.Tensor | None = None,
):
    """Build an `and_mask_function` for `create_causal_mask` from up to four
    constraints, AND-combined with causal by `create_causal_mask`:

    - dynamic eviction (`chunk_id [B,T]`, `evict_step [B,T]` int): the real
      training path. Each token belongs to a chunk; `evict_step[t]` is the chunk
      step at which that token's chunk is evicted to the RNN state. keep(q,kv) =
      `evict_step[kv] > chunk_id[q]` — key kv's chunk must still be live when
      query q's chunk is processed (Design X: RNN scans all, attention evicts).
    - static eviction (`is_evicted [B,T]` bool): the simple synthetic variant.
      keep(q,kv) = (kv live) OR (q evicted).
    - segments (`segment_ids [B,T]` int): keep(q,kv) = (kv shared, id==0) OR
      (same segment). PACK block-diagonal QA.
    - documents (`seq_ids [B,T]` int): keep(q,kv) = same document. cu_seqlens
      packing isolation.

    Returns None if no constraint is given. Works for flex_attention (BlockMask)
    and sdpa/eager (vmapped) backends.
    """
    preds = []
    if chunk_id is not None and evict_step is not None:
        preds.append(lambda b, q, kv: evict_step[b, kv] > chunk_id[b, q])
    if is_evicted is not None:
        preds.append(lambda b, q, kv: (~is_evicted[b, kv]) | is_evicted[b, q])
    if segment_ids is not None:
        preds.append(lambda b, q, kv: (segment_ids[b, kv] == 0) | (segment_ids[b, kv] == segment_ids[b, q]))
    if seq_ids is not None:
        preds.append(lambda b, q, kv: seq_ids[b, kv] == seq_ids[b, q])
    if not preds:
        return None

    def and_mask(b, h, q, kv):
        out = preds[0](b, q, kv)
        for p in preds[1:]:
            out = out & p(b, q, kv)
        return out

    return and_mask


# Backwards-compatible alias.
def make_evict_and_mask(is_evicted: torch.Tensor):
    return make_memory_and_mask(is_evicted)


class MemoryAttention(nn.Module):
    """Wraps the original Qwen3Attention (main path) and adds a GDN2 side branch.

    Per-forward memory control (`write_mask`, side-branch cache) is set as
    attributes by the model loop rather than threaded through kwargs, so the
    base attention path and HF decorators never see custom kwargs.
    """

    def __init__(self, base_attn: Qwen3Attention, config: MemoryQwen3Config) -> None:
        super().__init__()
        self.layer_idx = base_attn.layer_idx
        self.attn = base_attn  # main path: untouched Qwen3Attention
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
            layer_idx=base_attn.layer_idx,
            norm_eps=config.mem_norm_eps,
        )
        # per-forward control, set by MemoryQwen3Model.forward
        self._mem_write_mask: torch.Tensor | None = None
        self._mem_cache = None
        self._mem_use_cache: bool = False
        self._mem_cu_seqlens: torch.Tensor | None = None
        # Training diagnostics. Disabled by default; train/sft/loop.py enables
        # this on one layer so normal forward/inference pays no reduction cost.
        self._record_o_stats: bool = False
        self._last_o_stats: dict[str, torch.Tensor] = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs,
    ):
        o_main, attn_weights = self.attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        o_side, _, _ = self.mem(
            hidden_states,
            write_mask=self._mem_write_mask,
            past_key_values=self._mem_cache,
            use_cache=self._mem_use_cache,
            cu_seqlens=self._mem_cu_seqlens,
        )
        o_total = o_main + o_side
        if self._record_o_stats:
            with torch.no_grad():
                main_abs = o_main.detach().float().abs().mean()
                side_abs = o_side.detach().float().abs().mean()
                total_abs = o_total.detach().float().abs().mean()
                eps = torch.tensor(1e-12, device=total_abs.device, dtype=total_abs.dtype)
                self._last_o_stats = {
                    "main_out_abs": main_abs,
                    "side_out_abs": side_abs,
                    "total_out_abs": total_abs,
                    "side_out_ratio": side_abs / torch.maximum(total_abs, eps),
                    "side_main_ratio": side_abs / torch.maximum(main_abs, eps),
                }
        return o_total, attn_weights


class MemoryQwen3PreTrainedModel(Qwen3PreTrainedModel):
    config: MemoryQwen3Config
    _no_split_modules = ["Qwen3DecoderLayer"]


class MemoryQwen3Model(MemoryQwen3PreTrainedModel, Qwen3Model):
    config_class = MemoryQwen3Config

    def __init__(self, config: MemoryQwen3Config) -> None:
        Qwen3Model.__init__(self, config)
        # Replace self_attn with MemoryAttention on the selected layers.
        mem_layers = set(config.memory_layer_indices)
        for idx in mem_layers:
            layer = self.layers[idx]
            layer.self_attn = MemoryAttention(layer.self_attn, config)
        self._mem_layer_indices = sorted(mem_layers)
        self.post_init()

    def _set_mem_control(self, write_mask, mem_cache, mem_use_cache, cu_seqlens=None) -> None:
        for idx in self._mem_layer_indices:
            m = self.layers[idx].self_attn
            m._mem_write_mask = write_mask
            m._mem_cache = mem_cache
            m._mem_use_cache = mem_use_cache
            m._mem_cu_seqlens = cu_seqlens

    def _clear_mem_control(self) -> None:
        self._set_mem_control(None, None, False, None)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        is_evicted: torch.Tensor | None = None,
        write_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        seq_ids: torch.Tensor | None = None,
        chunk_id: torch.Tensor | None = None,
        evict_step: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        mem_cache=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0) + past_seen

        # RNN write gate (per token). An explicit `write_mask` always wins (PACK
            # freezes QA segments, while segment-write uses dual positions); otherwise Design Y
        # derives it from the eviction layout, and Design X scans continuously.
        if write_mask is not None:
            write_mask = write_mask.to(inputs_embeds.dtype)
        elif is_evicted is not None and self.config.memory_design == "Y":
            write_mask = is_evicted.to(inputs_embeds.dtype)

        # In RL mode, derive the serve-identical stride eviction schedule from
        # input_ids when the trainer passes no explicit schedule. Gated on
        # full-sequence forwards (use_cache off) — incremental decode never
        # sees the whole row and must not re-derive.
        rl_unpad_len: int | None = None
        if (
            getattr(self.config, "mem_stride_policy", False)
            and chunk_id is None and evict_step is None
            and input_ids is not None and not use_cache
        ):
            from .stride_glue import derive_stride_eviction
            chunk_id, evict_step, seq_ids_d = derive_stride_eviction(
                input_ids, attention_mask, position_ids)
            # Only a packed row (>=2 sequences) needs block-diagonal isolation; a
            # single-sequence row derives all-zero seq_ids -> leave seq_ids=None so
            # the mask predicate set (and the micro_bsz=1 numerics) are unchanged.
            if seq_ids is None and int(seq_ids_d.max()) > 0:
                seq_ids = seq_ids_d

            # Attention isolation alone does not stop the GDN scan across packed
            # rows; the side branch therefore needs per-sequence state resets.
            # Boundaries mirror the mask predicate exactly (seq_ids transitions).
            # B==1 covers the RL packed row (verl rmpad flattens the micro-batch
            # to [1, total_nnz] with attention_mask=None); multi-row packs come
            # from the SFT collator, which passes cu_seqlens explicitly.
            if (
                cu_seqlens is None and seq_ids is not None
                and input_ids.shape[0] == 1 and attention_mask is None
            ):
                sid = seq_ids[0]
                cut = (sid[1:] != sid[:-1]).nonzero().flatten() + 1
                cu_seqlens = torch.cat(
                    [sid.new_zeros(1), cut, sid.new_tensor([sid.numel()])]
                ).to(torch.int32)

            # Right-pad the [1, T] row to a bounded set of fixed shapes so flex
            # does not repeatedly compile new shapes. verl's rearrange_micro_batches treats the
            # token budget as soft (k=ceil(total/budget) bins, then balanced by
            # FLOPs, not tokens — short-seq bins overshoot), so rows above the
            # multiple do occur; those round up by a 1/8 quantum instead of
            # doubling to 2x the multiple, which bounds the additional shapes and
            # memory use. The pad span is a dedicated tail segment with its own seq_id
            # (attention cannot cross it), own cu_seqlens segment (GDN state
            # reset), inert chunk/evict; hidden states are sliced back to T
            # before returning, so the head/loss never see pad.
            pad_mult = getattr(self.config, "mem_rl_pad_multiple", 0) or 0
            if pad_mult:
                _T0 = inputs_embeds.shape[1]
                quantum = pad_mult if _T0 <= pad_mult else max(pad_mult // 8, 128)
                pad_len = -_T0 % quantum
            else:
                pad_len = 0
            if pad_len and inputs_embeds.shape[0] == 1 and attention_mask is None:
                rl_unpad_len = inputs_embeds.shape[1]
                if seq_ids is None:
                    seq_ids = torch.zeros_like(input_ids)
                if cu_seqlens is None:
                    cu_seqlens = torch.tensor(
                        [0, rl_unpad_len], dtype=torch.int32, device=input_ids.device)
                inputs_embeds = torch.cat(
                    [inputs_embeds,
                     inputs_embeds.new_zeros(1, pad_len, inputs_embeds.shape[2])], dim=1)
                position_ids = torch.cat(
                    [position_ids,
                     torch.arange(pad_len, device=position_ids.device).unsqueeze(0)], dim=1)
                chunk_id = torch.cat([chunk_id, chunk_id.new_zeros(1, pad_len)], dim=1)
                evict_step = torch.cat(
                    [evict_step, evict_step.new_full((1, pad_len), 1 << 30)], dim=1)
                seq_ids = torch.cat(
                    [seq_ids, seq_ids.new_full((1, pad_len), int(seq_ids.max()) + 1)], dim=1)
                cu_seqlens = torch.cat(
                    [cu_seqlens, cu_seqlens.new_tensor([inputs_embeds.shape[1]])])

        # Build the (eviction / segment / document-aware) causal mask, reused for
        # all layers. `seq_ids` isolates packed sequences (cu_seqlens path);
        # `chunk_id`/`evict_step` drive dynamic chunk eviction (real training).
        if not isinstance(attention_mask, dict):
            and_mask = make_memory_and_mask(is_evicted, segment_ids, seq_ids, chunk_id, evict_step)
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "and_mask_function": and_mask,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Set control on every forward (incl. None when no eviction), so there is
        # no stale state. We deliberately do NOT clear afterwards: gradient
        # checkpointing recomputes this forward during backward and must see the
        # same write_mask. Training is sequential (forward→backward→next forward),
        # so the values stay valid until the next forward overwrites them.
        self._set_mem_control(write_mask, mem_cache, bool(use_cache), cu_seqlens)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if rl_unpad_len is not None:
            hidden_states = hidden_states[:, :rl_unpad_len]
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MemoryQwen3ForCausalLM(MemoryQwen3PreTrainedModel, Qwen3ForCausalLM):
    config_class = MemoryQwen3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MemoryQwen3Config) -> None:
        # Build directly (don't call Qwen3ForCausalLM.__init__, which would
        # construct a throwaway base Qwen3Model first).
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = MemoryQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # Honor zero-init of the side o_proj for *both* construction paths
        # (post_init randomizes it, so this must run last). For from_qwen3 the
        # copy-init also zeros it; here it covers from-scratch construction.
        if config.mem_o_proj_zero_init:
            for idx in config.memory_layer_indices:
                nn.init.zeros_(self.model.layers[idx].self_attn.mem.o_proj.weight)
        # RL trainable specification: full-parameter side branch, all other parameters frozen.
        self._apply_freeze_main_path()

    def _apply_freeze_main_path(self) -> None:
        if getattr(self.config, "mem_freeze_main_path", False):
            for name, p in self.named_parameters():
                p.requires_grad = ".self_attn.mem." in name

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # meta-device loading rebuilds parameter tensors, dropping the
        # requires_grad set in __init__ — re-apply the freeze after load
        # (this is the path external trainers like verl take).
        model = super().from_pretrained(*args, **kwargs)
        model._apply_freeze_main_path()
        return model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        is_evicted: torch.Tensor | None = None,
        write_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        seq_ids: torch.Tensor | None = None,
        chunk_id: torch.Tensor | None = None,
        evict_step: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        mem_cache=None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            is_evicted=is_evicted,
            write_mask=write_mask,
            segment_ids=segment_ids,
            seq_ids=seq_ids,
            chunk_id=chunk_id,
            evict_step=evict_step,
            cu_seqlens=cu_seqlens,
            mem_cache=mem_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if labels is not None:
            # Answer-only logits: gather just the supervised positions and run
            # lm_head on those, so we never materialize [B, L, vocab] (≈40GB at
            # L=128k). Mathematically identical to full-seq CE with ignore_index
            # = -100 (mean over answer tokens -> normalized by answer-token count).
            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:].to(hidden_states.device)
            sel = shift_labels != -100
            sel_hidden = shift_hidden[sel]            # [n_answer, H] bf16
            sel_lab = shift_labels[sel]               # [n_answer]
            n = sel_hidden.shape[0]
            # Chunked lm_head + CE over the answer tokens: never materialize the
            # full [n_answer, vocab] fp32 logits (≈30GB when a 64k pack is mostly
            # answer, e.g. long open-ended replies -> OOM). sum/n == mean CE.
            if n == 0:
                loss = hidden_states.sum() * 0.0   # keep graph; no supervised token
            else:
                CH = 8192
                tot = hidden_states.new_zeros((), dtype=torch.float32)
                for s in range(0, n, CH):
                    lg = self.lm_head(sel_hidden[s:s + CH]).float()
                    tot = tot + nn.functional.cross_entropy(
                        lg, sel_lab[s:s + CH], reduction="sum")
                loss = tot / n
            return CausalLMOutputWithPast(loss=loss, logits=None,
                                          past_key_values=outputs.past_key_values)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
        )

    # ------------------------------------------------------------------ init
    @classmethod
    def from_qwen3(
        cls,
        qwen3_path: str,
        memory_design: str = "Y",
        mem_layers: list[int] | None = None,
        mem_o_proj_zero_init: bool = True,
        dtype: torch.dtype | None = torch.bfloat16,
        device_map: str | None = None,
        attn_implementation: str | None = None,
        **config_overrides,
    ) -> "MemoryQwen3ForCausalLM":
        """Build a MemoryQwen3 from a pretrained Qwen3: load base weights, then
        copy-init each side branch from the backbone attention geometry."""
        base = Qwen3ForCausalLM.from_pretrained(
            qwen3_path, dtype=dtype, attn_implementation=attn_implementation
        )
        config = MemoryQwen3Config(
            memory_design=memory_design,
            mem_layers=mem_layers,
            mem_o_proj_zero_init=mem_o_proj_zero_init,
            **{**base.config.to_dict(), **config_overrides},
        )
        if attn_implementation is not None:
            config._attn_implementation = attn_implementation

        model = cls(config)
        if dtype is not None:
            model = model.to(dtype=dtype)

        # 1) load all base weights that map directly (embed/mlp/norms/lm_head and,
        #    for wrapped layers, the main attention under `.self_attn.attn.*`).
        sd = _remap_base_state_dict(base.state_dict(), config.memory_layer_indices)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # side-branch params (model.layers.*.self_attn.mem.*) are expected-missing
        leftover = [k for k in missing if ".self_attn.mem." not in k]
        if leftover:
            raise RuntimeError(f"Unexpected missing keys after base load: {leftover[:8]} ...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading base: {unexpected[:8]} ...")

        # 2) copy-init each side branch from its (now-loaded) main attention.
        for idx in config.memory_layer_indices:
            mattn = model.model.layers[idx].self_attn
            copy_init_side_branch(mattn.mem, mattn.attn, config.mem_o_proj_zero_init)

        del base
        if device_map is not None:
            model = model.to(device_map)
        return model


def _remap_base_state_dict(state_dict: dict, mem_layers: list[int]) -> dict:
    """Insert `.attn` into self_attn keys for wrapped layers so base attention
    weights land on MemoryAttention.attn.*; all other keys pass through."""
    mem_set = set(mem_layers)
    sub = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    out = {}
    for k, v in state_dict.items():
        nk = k
        if ".self_attn." in k:
            parts = k.split(".")
            try:
                li = parts.index("layers")
                layer_idx = int(parts[li + 1])
            except (ValueError, IndexError):
                layer_idx = None
            if layer_idx in mem_set and any(f".self_attn.{s}." in k for s in sub):
                nk = k.replace(".self_attn.", ".self_attn.attn.", 1)
        out[nk] = v
    return out


@torch.no_grad()
def copy_init_side_branch(
    side: MemoryGatedDeltaNet2, attn: Qwen3Attention, zero_o: bool
) -> None:
    """Copy Qwen3 QKVO into the GDN2 side branch.

    Supports both the legacy full-MHA side branch (32 Q/K/V heads for Qwen3-4B)
    and the compact KV-head branch (8 Q/K/V heads + expanded V):
      - Q: direct copy if head counts match; if target heads match backbone KV
        heads, average the corresponding GQA Q group.
      - K: copy/adapt from backbone KV heads.
      - V: copy/adapt from backbone KV heads, then block-repeat each V head along
        its channel dimension when `expand_v > 1`.
      - O: copied only when shapes match; normally zero-initialized for training.
    """
    hd = attn.head_dim
    dt = side.q_proj.weight.dtype
    qh = attn.q_proj.weight.shape[0] // hd
    kvh = attn.k_proj.weight.shape[0] // hd

    def adapt_heads(heads: torch.Tensor, target_heads: int, name: str) -> torch.Tensor:
        src_heads = heads.shape[0]
        if target_heads == src_heads:
            return heads
        if target_heads > src_heads and target_heads % src_heads == 0:
            return heads.repeat_interleave(target_heads // src_heads, dim=0)
        if src_heads > target_heads and src_heads % target_heads == 0:
            return heads.view(target_heads, src_heads // target_heads, hd, -1).mean(dim=1)
        raise ValueError(f"cannot adapt {name} heads from {src_heads} to {target_heads}")

    q_heads = attn.q_proj.weight.view(qh, hd, -1)
    if side.num_heads == qh:
        q_init = q_heads
    elif qh % kvh == 0 and side.num_heads == kvh:
        q_init = q_heads.view(kvh, qh // kvh, hd, -1).mean(dim=1)
    else:
        q_init = adapt_heads(q_heads, side.num_heads, "q")
    side.q_proj.weight.copy_(q_init.reshape(side.q_proj.weight.shape).to(dt))

    k_heads = attn.k_proj.weight.view(kvh, hd, -1)
    k_init = adapt_heads(k_heads, side.num_heads, "k")
    side.k_proj.weight.copy_(k_init.reshape(side.k_proj.weight.shape).to(dt))

    v_heads = adapt_heads(attn.v_proj.weight.view(kvh, hd, -1), side.num_v_heads, "v")
    if side.head_v_dim % hd != 0:
        raise ValueError(
            f"side.head_v_dim={side.head_v_dim} must be a multiple of backbone head_dim={hd} "
            "for copy initialization"
        )
    v_expand = side.head_v_dim // hd
    v_init = v_heads.repeat(1, v_expand, 1)
    side.v_proj.weight.copy_(v_init.reshape(side.v_proj.weight.shape).to(dt))

    if zero_o:
        side.o_proj.weight.zero_()
    else:
        if side.o_proj.weight.shape != attn.o_proj.weight.shape:
            raise ValueError(
                f"cannot copy-init o_proj with shape {tuple(side.o_proj.weight.shape)} "
                f"from backbone shape {tuple(attn.o_proj.weight.shape)}; use zero_o=True"
            )
        side.o_proj.weight.copy_(attn.o_proj.weight.to(side.o_proj.weight.dtype))
