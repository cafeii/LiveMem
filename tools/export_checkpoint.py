"""Export a trainable LiveMem checkpoint as a complete Hugging Face checkpoint.

The default recipe loads the full recurrent memory branch from ``trainable.pt``
onto the frozen base model. Optional LoRA checkpoints are detected from their
weight names and folded automatically before export.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from models.modeling_memory_qwen3 import MemoryQwen3ForCausalLM  # noqa: E402
from train.sft.param_groups import apply_trainable_spec  # noqa: E402

QWEN3 = os.environ.get("QWEN3", "Qwen/Qwen3-4B-Instruct-2507")
STEP = os.environ.get("STEP", "outputs/sft/LiveMem-SFT-stage2/step_500")
OUT = os.environ.get("OUT", "outputs/sft/LiveMem-SFT")
# Memory-branch geometry overrides for compact-head/expand_v variants; match training YAML.
GEOM = {k: t(os.environ[e]) for e, k, t in [
    ("MEM_NUM_HEADS", "mem_num_heads", int),
    ("MEM_NUM_V_HEADS", "mem_num_v_heads", int),
    ("MEM_EXPAND_V", "mem_expand_v", float),
] if e in os.environ}


def main():
    # Construct in fp32 so an optional adapter can be folded without losing its
    # small update to bf16 rounding; the final checkpoint is saved in bf16.
    device = os.environ.get("DEVICE", "cuda")  # Set DEVICE=cpu when training occupies the GPU.
    if GEOM:
        print("geometry overrides:", GEOM)
    trn = torch.load(os.path.join(STEP, "trainable.pt"), map_location="cpu",
                     weights_only=False)
    is_lora = any("lora_" in k for k in trn)
    print(f"trainable.pt: {len(trn)} tensors, spec={'rnn=lora' if is_lora else 'rnn=full'}")

    base_merged = os.environ.get("BASE_MERGED")
    if base_merged:
        # init_from_merged training starts from a complete merged model, so rebuilding
        # must use the same checkpoint; from_qwen3 would zero-initialize the branch again.
        print("base = from_pretrained:", base_merged)
        model = MemoryQwen3ForCausalLM.from_pretrained(
            base_merged, dtype=torch.float32, attn_implementation="eager")
    else:
        model = MemoryQwen3ForCausalLM.from_qwen3(
            QWEN3, memory_design="X", mem_layers=None, mem_o_proj_zero_init=True,
            dtype=torch.float32, attn_implementation="eager", **GEOM)
    if is_lora:
        # SPEC looks like "rnn=lora,attn=lora,ffn=lora"; by default only GDN uses LoRA.
        spec = dict(kv.split("=") for kv in os.environ.get(
            "SPEC", "rnn=lora,attn=frozen,ffn=frozen").split(","))
        lora_cfg = {"r": int(os.environ.get("LORA_R", 64)),
                    "alpha": int(os.environ.get("LORA_ALPHA", 128)), "dropout": 0.0}
        model, summ = apply_trainable_spec(model, spec, lora_cfg=lora_cfg)
        print("spec:", summ["spec"], "trainable%:", round(summ["pct"], 4))

    missing, unexpected = model.load_state_dict(trn, strict=False)
    print(f"loaded {len(trn)} trained tensors; unexpected={len(unexpected)} (want 0); "
          f"missing={len(missing)} (frozen base, expected large)")
    assert len(unexpected) == 0, ("unexpected keys", unexpected[:8])

    model = model.to(device).eval()
    if device == "cpu":
        # The GDN branch uses an FLA Triton kernel and cannot run on CPU. Export does
        # not require a forward pass, so validate unexpected==0 and finite weights.
        merged = model.merge_and_unload().eval() if is_lora else model
        for k, v in merged.state_dict().items():
            if ".mem." in k:
                assert torch.isfinite(v).all(), f"non-finite weight: {k}"
        print("cpu export: forward sanity skipped (fla Triton is GPU-only); "
              "mem.* weights all finite")
    else:
        ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], device=device)
        with torch.inference_mode():
            lg_peft = model(input_ids=ids).logits.float()
        assert torch.isfinite(lg_peft).all(), "non-finite logits after load"
        if is_lora:
            merged = model.merge_and_unload().eval()
            with torch.inference_mode():
                lg_merged = merged(input_ids=ids).logits.float()
            diff = (lg_peft - lg_merged).abs().max().item()
            print(f"merge sanity: max|logit diff (peft vs merged)| = {diff:.2e} (want ~0)")
        else:
            merged = model  # rnn=full has no LoRA weights to fold and is ready after loading.
            print("rnn=full: no LoRA fold; forward sanity passed (finite logits)")

    os.makedirs(OUT, exist_ok=True)
    merged = merged.to(torch.bfloat16)  # serve in bf16
    merged.save_pretrained(OUT, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(QWEN3).save_pretrained(OUT)
    nmem = sum(1 for k in merged.state_dict() if ".mem." in k)
    print(f"saved merged ckpt -> {OUT}  (mem.* tensors={nmem})")


if __name__ == "__main__":
    main()
