<div align="center">

# LiveMem

**Maintaining Memory State Continuity in Long-Running LLM Inference**

[Paper](https://arxiv.org/pdf/2608.02515) ·
[LiveMem-SFT](https://huggingface.co/chen-l/LiveMem-SFT) ·
[LiveMem-RL](https://huggingface.co/chen-l/LiveMem-RL)

English · [中文](README_zh.md)

![LiveMem overview](assets/livemem.png)

<div align="left">

## Overview

Long-running assistants and agents eventually face **context turnover**: their working context has a finite capacity, but the model still needs to carry computation forward after earlier context leaves the active window. LiveMem formulates this capability as **state continuity under context turnover** and maintains historical information in a fixed-capacity memory state after old context is evicted from attention.

LiveMem adds a parallel Gated DeltaNet-2 (GDN2) recurrent memory branch to every attention layer of a pretrained Qwen3 model:

- The attention path retains the system prompt and a bounded recent KV window for precise short-term access.
- The recurrent memory path continuously reads and updates a fixed-capacity memory state throughout the lifetime of a request.
- Their outputs are added at every layer: `o = o_attention + o_memory`.
- Training uses dynamic attention masks to simulate context turnover, while inference releases evicted KV pages, preserving the same visibility boundary in training and deployment.
- Memory-oriented SFT and RL require the model to rely on the memory state after supporting evidence has left the attention window.

![LiveMem architecture](assets/structure.jpg)

### `state` and `truncate`

The repository uses `mode` and `window_size` to describe how long histories are handled:

| Mode | History handling | Meaning of `window_size` |
|---|---|---|
| `state` | The full history is continuously written into the recurrent state; attention retains only the system sink and the recent window | Number of live tokens retained by attention; the current serving script supports 8K or 32K |
| `truncate` | Older history is discarded before the model sees it, and the retained suffix is processed from a zero state | Number of memory tokens retained after truncation |

A single input to the current LiveMem checkpoints is still bounded by a 262,144-token positional horizon. The memory state persists within one request and is released when that request finishes. The current OpenAI-compatible API does not automatically reuse the state across independent requests.

## Usage

### 1. Environment setup

```bash
pip install -r requirements.txt
pip install -e "third_party/flash-linear-attention[cuda]"
```

### 2. Hugging Face quick check

The following example creates a separate GDN2 cache explicitly so the memory state is preserved between prefill and token-by-token decoding. This path is intended for model loading and functional checks. It does not perform the paged-KV window eviction implemented by the vLLM integration; use the vLLM service in the next section for windowed `state` inference.

```python
import torch

from models import MemoryQwen3ForCausalLM  # Import models first to load the vendored FLA.
from fla.models.utils import Cache as FLACache
from transformers import AutoTokenizer

model_id = "chen-l/LiveMem-RL"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = MemoryQwen3ForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    attn_implementation="eager",
).eval().to("cuda")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Remember that my project codename is Aurora. What is the codename?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

memory_cache = FLACache()
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        mem_cache=memory_cache,
        use_cache=True,
        max_new_tokens=128,
        do_sample=False,
    )

print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
```

### 3. OpenAI-compatible serving with vLLM

The LiveMem serving plugin targets `vllm==0.19.1`. Install vLLM, FlashInfer, and the repository plugin in a separate vLLM environment:

```bash
pip install vllm==0.19.1 flashinfer-python
pip install -e models/vllm
```

Start a `state` server with a 32K live attention window:

```bash
CKPT=chen-l/LiveMem-RL \
MODE=state WINDOW_SIZE=32768 \
GPUS=0 PORT=8821 \
bash scripts/serve_livemem.sh
```

Call the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8821/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "memq_state-32k",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "{memory_docs}\n\nSummarize the supplied long history."}
    ],
    "temperature": 0,
    "max_tokens": 512
  }'
```

Use `MODE=state WINDOW_SIZE=8192` for an 8K state window. A `truncate` server can be started with `MODE=truncate WINDOW_SIZE=<tokens>`, but truncation is performed by the client. The client should preserve the instruction, query, and generation budget, remove content only from the oldest end of the memory, and send the most recent `window_size` memory tokens to the server.

## Reproduction

This section lists the repository entry points for data processing, two-stage SFT, checkpoint export, RL, and evaluation. The paper uses Qwen3 Instruct as the backbone. Data scripts read `TOKENIZER_PATH`, checkpoint export reads `QWEN3`, and SFT accepts a Hugging Face ID or local path through `--set model.qwen3_path=<path>`. In the commands below, `BASE_MODEL` denotes that backbone ID or local directory.

### 1. Prepare training data

```bash
export BASE_MODEL="/path/to/qwen3-instruct"

# Download the training datasets used in the paper to dataset/raw/.
python scripts/download_datasets.py --group train

# Convert them to the unified memory/QA schema.
python tools/data_process/run_preprocess.py --group train --jobs 10
python tools/data_process/ttl_assemble.py
python tools/data_process/split.py

# Pre-tokenize the data and write dataset/train/<dataset>/*.arrow.
TOKENIZER_PATH="$BASE_MODEL" \
python tools/data_process/tokenize_train.py --nproc 32
```

Run `python scripts/download_datasets.py --list` to inspect the complete dataset list. The evaluation download group includes InfiniteBench and MemoryAgentBench.

### 2. Two-stage SFT

The paper's SFT recipe uses two nodes with eight A800-80GB GPUs per node, BF16, and fixed 64K-token packs. Set `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` on each node, then run the same command on both nodes.

Stage 1 trains for 100 steps on LongAlign, LongAlpaca, LongMIT, and Long-Data-Collections:

```bash
PYTHONPATH=. torchrun \
  --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  train/sft/train_sft.py \
  --config train/sft/config/livemem_sft_stage1.yaml \
  --set model.qwen3_path="$BASE_MODEL"
```

Stage 2 resumes from Stage 1's `step_100/trainable.pt` and trains for 500 steps on a mixture of QA, classification, multi-question, and long-form data:

```bash
PYTHONPATH=. torchrun \
  --nnodes=2 --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  train/sft/train_sft.py \
  --config train/sft/config/livemem_sft_stage2.yaml \
  --set model.qwen3_path="$BASE_MODEL"
```

If the Stage 1 output directory changes, override the Stage 2 resume path with `--set train.resume_trainable_from=<path>`. Environments with different memory or node counts may adjust pack size, gradient accumulation, and training steps.

### 3. Export the SFT checkpoint

Write the fully trained memory branch back onto the frozen backbone and export a complete checkpoint loadable by Hugging Face and vLLM:

```bash
QWEN3="$BASE_MODEL" \
STEP=outputs/sft/LiveMem-SFT-stage2/step_500 \
OUT=outputs/sft/LiveMem-SFT \
PYTHONPATH=. python tools/export_checkpoint.py
```

Set `DEVICE=cpu` to export weights on CPU. This skips the forward sanity check because it depends on Triton.

### 4. RL

First convert the RL splits to the verl data format:

```bash
TOKENIZER_PATH="$BASE_MODEL" \
python tools/data_process/to_verl_parquet.py
```

`run_grpo_dev.sh` is the single-node development and validation entry point. It uses GRPO, eight responses per group, no KL penalty, 0.20/0.28 clipping, token-mean loss, and asynchronous vLLM rollouts:

```bash
CKPT=outputs/sft/LiveMem-SFT \
DATA=dataset/train/rl_verl/mix_bal_v1.parquet \
VAL=dataset/train/rl_verl/mix_bal_v1_val.parquet \
RUN=LiveMem-RL CUDA_VISIBLE_DEVICES=0,1 NGPUS=2 STEPS=5 \
bash scripts/rl/run_grpo_dev.sh
```

The paper-scale setup uses two 8-GPU trainer nodes, one 8-GPU rollout node, and a separate 2-GPU judge service. To scale to that setup, adapt `trainer.nnodes`, the Ray cluster, and rollout/judge resource placement to the target cluster.

### 5. Evaluation

Prepare the evaluation data:

```bash
python scripts/download_datasets.py --group test
python tools/data_process/run_preprocess.py --group test --jobs 8
python tools/data_process/eval_split.py
```

Start the `truncate`, `state-32k`, and `state-8k` LiveMem server pools and the judge server in separate terminals or on separate nodes:

```bash
CKPT=outputs/sft/LiveMem-SFT MODE=truncate WINDOW_SIZE=262144 GPUS=0 PORT=8811 bash scripts/serve_livemem.sh
CKPT=outputs/sft/LiveMem-SFT MODE=state WINDOW_SIZE=32768 GPUS=1 PORT=8821 bash scripts/serve_livemem.sh
CKPT=outputs/sft/LiveMem-SFT MODE=state WINDOW_SIZE=8192 GPUS=2 PORT=8831 bash scripts/serve_livemem.sh
MODEL=<judge-checkpoint> GPUS=3,4 PORT=8790 bash scripts/serve_judge.sh
```

Run the main LiveMem evaluation matrix:

```bash
SERVERS='{"truncate":"http://127.0.0.1:8811/v1","state-32k":"http://127.0.0.1:8821/v1","state-8k":"http://127.0.0.1:8831/v1"}' \
JUDGE_URL=http://127.0.0.1:8790/v1 \
TOKENIZER_PATH="$BASE_MODEL" \
PY=python \
bash scripts/eval/run_livemem.sh
```

Evaluation supports resumable execution and writes to `results/eval/` by default. Serving and evaluation entry points for the other baselines are available under `scripts/serve_*.sh` and `scripts/eval/run_*.sh`.

### Optional LoRA support

The default LiveMem-SFT and LiveMem-RL recipes fully train the recurrent memory branch while freezing the attention and FFN main path; they do not use LoRA. The training framework can optionally enable LoRA by parameter group, and the export tool automatically detects and folds optional LoRA adapters.

## Citation

```bibtex
@article{liu2026livemem,
  title   = {LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference},
  author  = {Zhichen Liu and Ruihan Sun and Hengjie Yang and Zipeng Wu and Zhaohan Chen and Xiaofan Zhang and Yang Xu},
  journal = {arXiv preprint arXiv:2608.02515},
  year    = {2026}
}
```
