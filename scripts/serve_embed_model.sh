#!/usr/bin/env bash
# Qwen3-Embedding OpenAI-compatible pooling server.
set -euo pipefail

PY=${PY:-python}
GPU=${GPU:-0}
PORT=${PORT:-8811}
HOST=${HOST:-0.0.0.0}
MODEL=${MODEL:-/nas/lzc/model/Qwen3-Embedding-0.6B}
UTIL=${UTIL:-0.20}
MAXLEN=${MAXLEN:-32768}

export PYTHONUNBUFFERED=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

echo "[serve_embedding] gpu=$GPU port=$PORT model=$MODEL maxlen=$MAXLEN"
CUDA_VISIBLE_DEVICES="$GPU" exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name qwen3-embedding \
  --runner pooling --convert embed --dtype bfloat16 \
  --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
  --max-num-seqs 128 --max-num-batched-tokens 65536 \
  --host "$HOST" --port "$PORT"
