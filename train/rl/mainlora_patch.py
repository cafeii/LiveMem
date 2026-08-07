"""Train the full side branch together with main-path LoRA under verl.

verl's LoRA path (`FSDPEngineWithLMHead._build_lora_module`) wraps the model
with peft, which freezes every base parameter, while this configuration needs the GDN
side branch (`.self_attn.mem.`, incl. non-Linear A_log/dt_bias/conv/o_norm)
trained FULL-param alongside the main-path LoRA adapters. peft's
``modules_to_save`` is not usable here because verl does not
plumb it, its `modules_to_save.default.*` state-dict keys survive
`normalize_peft_param_name` and would KeyError vllm's load_weights on sync,
and it duplicates every wrapped module (frozen copy + trainable copy).

Fix = unfreeze the side-branch BASE parameters right after get_peft_model:
- ordering is safe: _build_lora_module (unfreeze here) -> _build_fsdp_module
  (FSDP2 supports mixed requires_grad natively) -> _build_optimizer
  (build_optimizer takes module.parameters() unfiltered, so the unfrozen
  params enter the optimizer; frozen params get grad=None and AdamW skips).
- weight sync REQUIRES `actor_rollout_ref.model.lora.merge=True`: the merge
  branch of get_per_tensor_param ships the full state_dict (side-branch keys
  are clean HF names after normalize_peft_param_name); the adapter-delta
  branch (merge=False) only collects lora_A/B and would silently DROP the
  side-branch updates (transformer_impl.py:806-822).
- LoRA targets must be a regex hitting ONLY the main path: `.mem` has
  same-named q/k/v/o_proj Linears, so bare-suffix targets would hit both.
  The script targets only main-path projection layers.

Inert when LoRA is off (lora_rank=0 skips _build_lora_module entirely) or on
models without a side branch. Installed from memory_hf_registry (external_lib,
imported in the WorkerDict before model build). Idempotent.
"""
from __future__ import annotations


def install() -> None:
    try:
        from verl.workers.engine.fsdp import transformer_impl as ti
    except Exception as e:  # pragma: no cover
        print(f"[mainlora-patch] verl import failed, not patching: {e}", flush=True)
        return
    cls = ti.FSDPEngineWithLMHead
    if getattr(cls, "_mem_mainlora_patched", False):
        return
    orig = cls._build_lora_module

    def _build_lora_module(self, module):
        module = orig(self, module)
        n = numel = 0
        for name, p in module.named_parameters():
            if ".self_attn.mem." in name and not p.requires_grad:
                p.requires_grad_(True)
                n += 1
                numel += p.numel()
        if n:
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"[mainlora-patch] unfroze {n} side-branch tensors "
                  f"({numel / 1e9:.3f}B params); total trainable {trainable / 1e9:.3f}B",
                  flush=True)
        # peft's autocast_adapter_dtype=True default upcasts lora_A/B to fp32; with
        # fsdp_config.model_dtype=bf16 that mixes {bf16, fp32} in one decoder-layer
        # shard group and trips fsdp2's "uniform original parameter dtype" assert.
        # Cast adapters back to the base dtype used by the complete model.
        base_dtype = next(p.dtype for name, p in module.named_parameters()
                          if "lora_" not in name)
        cast = 0
        for name, p in module.named_parameters():
            if "lora_" in name and p.dtype != base_dtype:
                p.data = p.data.to(base_dtype)
                cast += 1
        if cast:
            print(f"[mainlora-patch] cast {cast} LoRA tensors to {base_dtype} "
                  "(uniform dtype for fsdp2)", flush=True)
        return module

    cls._build_lora_module = _build_lora_module
    cls._mem_mainlora_patched = True
    print("[mainlora-patch] patched FSDPEngineWithLMHead._build_lora_module "
          "(side-branch full-param unfreeze after peft wrap)", flush=True)
