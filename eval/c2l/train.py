"""Train a Context2LoRA adapter for one memory.

Official settings from narrativeqa/logic/trainer.py and
12_generate_final_multilora_configs.py:
- r=4, alpha=8, lr=5e-4, steps=150, batch size 32, warmup 0.1, dropout 0.1,
  and no bias.
- ``target_modules="all-linear"``.
- Each sample is ``[{user: q}, {assistant: a}]`` passed through the chat
  template with EOS appended. All non-padding labels are trained when
  ``mask_question=False``.
- The step count is fixed independently of dataset size; the data loader cycles
  indefinitely until reaching `steps`.

Difference from upstream: the official code masks labels by `pad_token_id`,
which also masks sentence-final EOS when pad equals EOS. This implementation
masks padding with `attention_mask`, preserving the intended semantics without
dropping EOS.
"""
from __future__ import annotations

import os
import random


def train_lora(base_model, tokenizer, qa_pairs: list[dict], out_dir: str,
               r: int = 4, alpha: int = 8, lr: float = 5e-4, steps: int = 150,
               bs: int = 32, max_len: int = 2048, warmup_ratio: float = 0.1,
               seed: int = 42, mask_question: bool = False):
    """Train LoRA in place on a loaded base model and save it to `out_dir`.

    Returns a PeftModel. `get_peft_model` injects LoRA layers in place and wraps
    `base_model`; callers must restore the base with
    ``base = peft_model.unload()`` after use to release adapter memory and
    restore clean weights.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW
    from transformers import get_scheduler

    device = next(base_model.parameters()).device
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.1, bias="none",
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(base_model, cfg)
    model.train()

    # Build sample text like the official trainer._collate_fn: apply the chat
    # template and append EOS. With mask_question=True, also record prompt
    # length including the generation header and retain labels only for the
    # answer. ICL direct SFT uses arbitrary numeric-label mappings, so full-
    # sequence loss would be diluted by utterance modeling.
    texts, plens = [], []
    for qa in qa_pairs:
        q = qa.get("q") or qa.get("question")
        a = qa.get("a") or qa.get("answer")
        if not q or not a:
            continue
        msgs = [{"role": "user", "content": str(q)}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        t = tokenizer.apply_chat_template(
            msgs + [{"role": "assistant", "content": str(a)}],
            tokenize=False, add_generation_prompt=False)
        if not t.endswith(tokenizer.eos_token):
            t += tokenizer.eos_token
        texts.append(t)
        plens.append(len(tokenizer.encode(prompt, add_special_tokens=False))
                     if mask_question else 0)
    if not texts:
        raise ValueError("qa_pairs 为空或格式不合法，无法训练")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=lr)
    sched = get_scheduler("cosine", optim,
                          num_warmup_steps=int(steps * warmup_ratio),
                          num_training_steps=steps)

    # Cycle indefinitely to a fixed number of steps, independent of dataset
    # size as in the official protocol, and shuffle within each epoch.
    rng = random.Random(seed)
    order: list[int] = []
    for step in range(steps):
        batch, bplens = [], []
        for _ in range(bs):
            if not order:
                order = list(range(len(texts)))
                rng.shuffle(order)
            i = order.pop()
            batch.append(texts[i])
            bplens.append(plens[i])
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        if mask_question:  # With right padding, the first `plen` tokens are the prompt.
            for i, pl in enumerate(bplens):
                labels[i, :pl] = -100
        loss = model(**enc, labels=labels).loss
        if torch.isnan(loss):
            raise RuntimeError(f"step {step}: loss NaN")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    del optim, sched
    torch.cuda.empty_cache()
    return model
