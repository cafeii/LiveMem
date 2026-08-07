"""verl ``model.external_lib`` hook for the memory model.

verl imports this module before building config/model
(`HFModelConfig.__post_init__` -> `import_external_libs`), so registering the
memory model into HF Auto* here lets the stock FSDP engine load it:
  * AutoConfig.from_pretrained(ckpt)  -> MemoryQwen3Config  (model_type match)
  * get_hf_auto_model_class           -> AutoModelForCausalLM ("...ForCausalLM"
    architecture suffix)
  * AutoModelForCausalLM.from_pretrained(ckpt, config=...) -> MemoryQwen3ForCausalLM

Requires the workspace root on sys.path/PYTHONPATH (packages `models`, `train`).
Idempotent (exist_ok / re-register is a no-op with the same classes).
"""
from transformers import AutoConfig, AutoModelForCausalLM

from models.configuration_memory_qwen3 import MemoryQwen3Config
from models.modeling_memory_qwen3 import MemoryQwen3ForCausalLM
from train.rl.finite_loss_patch import install as _install_finite_loss_patch
from train.rl.flash_attn_shim import install as _install_flash_attn_shim
from train.rl.mainlora_patch import install as _install_mainlora_patch

_install_flash_attn_shim()
_install_finite_loss_patch()
_install_mainlora_patch()  # Inert unless lora_rank > 0.

AutoConfig.register("memory_qwen3", MemoryQwen3Config, exist_ok=True)
AutoModelForCausalLM.register(MemoryQwen3Config, MemoryQwen3ForCausalLM, exist_ok=True)
