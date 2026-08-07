#!/usr/bin/env bash
# Dense RAG orchestration server. Chunking/top-k exactly match the BM25 server.
set -euo pipefail

WS=${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${PY:-/usr/bin/python3}
PORT=${PORT:-8891}
N=${N:-1}
: "${GEN_URL:?set GEN_URL to the remote Qwen3-4B OpenAI URL pool}"
EMBED_URL=${EMBED_URL:-http://localhost:8811/v1}
WINDOW=${WINDOW:-512}
THRESHOLD=${THRESHOLD:-20}
CACHE_SIZE=${CACHE_SIZE:-4}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-64}
INDEX_CONCURRENCY=${INDEX_CONCURRENCY:-4}
TOK_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}

export PYTHONUNBUFFERED=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

cd "$WS"
pids=()
for ((i = 0; i < N; i++)); do
  port=$((PORT + i))
  echo "[serve_embedding_rag] instance$i port=$port gen=$GEN_URL embed=$EMBED_URL window=$WINDOW"
  "$PY" -m eval.embedding_rag_server --port "$port" --gen-url "$GEN_URL" \
    --embed-url "$EMBED_URL" --tok-path "$TOK_PATH" --window "$WINDOW" \
    --threshold "$THRESHOLD" --cache-size "$CACHE_SIZE" \
    --embed-batch-size "$EMBED_BATCH_SIZE" --index-concurrency "$INDEX_CONCURRENCY" &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
wait
