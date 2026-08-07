"""GDN2 memory side-branch with an optional per-token write gate.

This subclasses fla's `GatedDeltaNet2` and overrides `forward` to apply a
`write_mask` that freezes the recurrence on read tokens. The GDN-2 update is

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T

so setting g_t = b_t = w_t = 0 gives S_t = S_{t-1} (read-only); the output
o_t = q_t · S_t is still produced (the token reads the memory). Verified
against `fla.ops.gdn2.naive.naive_recurrent_gdn2`.

  - Design Y: write_mask = is_evicted (open the gate only on the compress/evict
    region; live + Q + A are frozen read-only).
  - Design X: write_mask = None -> standard continuous scan over all tokens.
"""
from __future__ import annotations

from . import _bootstrap  # noqa: F401  (must run before importing fla)

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.layers.gdn2 import GatedDeltaNet2
from fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    update_layer_cache,
)
from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2


class MemoryGatedDeltaNet2(GatedDeltaNet2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        write_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        """Mirrors fla.GatedDeltaNet2.forward, adding `write_mask` support.

        `write_mask`: float/bool [batch, seq_len]; 0 freezes the recurrence
        (g=b=w=0 -> S_t = S_{t-1}). Padding (`attention_mask`) + `write_mask`
        together are not supported (the unpad reorder would desync the mask);
        the training path always passes `attention_mask=None`.
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a [batch_size, seq_len] 0/1 padding mask."
            )
            assert write_mask is None, (
                "write_mask is incompatible with a 2D padding mask (unpad reorder)."
            )

        cu_seqlens = kwargs.get("cu_seqlens")
        batch_size, q_len, _ = hidden_states.shape
        if cu_seqlens is not None and cu_seqlens.ndim == 2:
            if use_cache:
                raise ValueError("batched cu_seqlens training path does not support cache")
            outs = []
            for b in range(batch_size):
                cu = cu_seqlens[b]
                cu = cu[cu >= 0].contiguous()
                wm = write_mask[b:b + 1] if write_mask is not None else None
                o, _, _ = self.forward(
                    hidden_states[b:b + 1],
                    write_mask=wm,
                    attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=output_attentions,
                    cu_seqlens=cu,
                )
                outs.append(o)
            return torch.cat(outs, dim=0), None, past_key_values

        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)
        indices = None
        if cu_seqlens is None and attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), cache=conv_state_q,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), cache=conv_state_k,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), cache=conv_state_v,
                output_final_state=use_cache, cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
        g = -self.A_log.float().exp().unsqueeze(-1) * g

        # --- write gate (the only addition over fla) -----------------------
        # Zeroing g/b/w on read tokens freezes the state at S_{t-1}.
        if write_mask is not None:
            wm = write_mask.to(g.dtype).view(write_mask.shape[0], write_mask.shape[1], 1, 1)
            g = g * wm
            b = b * wm.to(b.dtype)
            w = w * wm.to(w.dtype)
        # -------------------------------------------------------------------

        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g, b)
            )

        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_gdn2(
                q=q, k=k, v=v, g=g, b=b, w=w,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gdn2(
                q=q, k=k, v=v, g=g, b=b, w=w,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Unsupported mode `{mode}`.")

        update_layer_cache(
            self, past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
