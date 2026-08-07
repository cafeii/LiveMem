#!/usr/bin/env bash
# Qwen3.6-35B-A3B judge server for the yes/no scoring workers in eval/pipeline.py.
# For reasoning models, scoring requests set
# chat_template_kwargs.enable_thinking=False.
# GPU budget: ~70 GB in bf16, so one TP=2 instance fits on 2 x 80 GB GPUs.
# Usage: GPUS=0,1 PORT=8790 bash scripts/serve_judge.sh
#   -> 1 instance on :8790. Pass http://IP:8790/v1 to --judge-url; use a
#      comma-separated list to round-robin across multiple instances.
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-python}   # Requires the vLLM 0.19.1 environment described in the README.
GPUS=${GPUS:-0,1}
PORT=${PORT:-8790}
TP=${TP:-2}
UTIL=${UTIL:-0.90}
MODEL=${MODEL:-/nas/lzc/model/Qwen3.6-35B-A3B}

export PYTHONUNBUFFERED=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
pids=()
idx=0
for ((i = 0; i + TP <= ${#GPU_ARR[@]}; i += TP)); do
  devs=$(IFS=,; echo "${GPU_ARR[*]:i:TP}")
  port=$((PORT + idx))
  echo "[serve_judge] instance$idx gpus=$devs port=$port"
  CUDA_VISIBLE_DEVICES=$devs "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name judge \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$UTIL" \
    --host 0.0.0.0 --port "$port" &
  pids+=("$!")
  idx=$((idx + 1))
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
