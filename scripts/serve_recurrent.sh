#!/usr/bin/env bash
# OpenAI-compatible, CPU-only recurrent-LLM baseline server. It summarizes
# each chunk into memory blocks, then answers from those blocks while forwarding
# generation to a reusable base-vLLM pool (start serve_qwen3_4b.sh first).
# A larger read window requires fewer rounds and lowers cost. Blocks are cached
# by memory hash and reused within each group.
# Usage: GEN_URL=http://localhost:8801/v1,http://localhost:8802/v1 PORT=8901 N=1 bash scripts/serve_recurrent.sh
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-/usr/bin/python3}   # Requires an HF tokenizer environment, not vLLM.
PORT=${PORT:-8901}
N=${N:-1}
: "${GEN_URL:?set GEN_URL to a comma-separated base-vLLM URL pool}"
READ_WINDOW=${READ_WINDOW:-16384}
BLOCK_CAP=${BLOCK_CAP:-8192}
CACHE_SIZE=${CACHE_SIZE:-512}
TOK_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}

export PYTHONUNBUFFERED=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

cd "$WS"
pids=()
for ((i = 0; i < N; i++)); do
  port=$((PORT + i))
  echo "[serve_recurrent] instance$i port=$port gen=$GEN_URL read_window=$READ_WINDOW cache_size=$CACHE_SIZE"
  "$PY" -m eval.recurrent_server --port "$port" --gen-url "$GEN_URL" \
    --tok-path "$TOK_PATH" --read-window "$READ_WINDOW" --block-cap "$BLOCK_CAP" \
    --cache-size "$CACHE_SIZE" &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
