#!/usr/bin/env bash
# Evaluate the recurrent baseline on the core matrix. It summarizes each chunk
# into memory blocks, then answers from those blocks.
#   SERVERS='{"truncate":"http://H:8901/v1,..."}' JUDGE_URL=... bash scripts/eval/run_recurrent.sh
# Server: GEN_URL=<Qwen3-4B vLLM pool> bash scripts/serve_recurrent.sh
# Only core single-example datasets are evaluated. Block construction is
# serialized within each group (one lock per memory) and parallelized across
# groups. d_high admission counts individual requests, so GROUP_CAP=10 splits
# large groups. Per-hash locks deduplicate blocks, although the same memory may
# still be built once per server. DHIGH=512 controls deep admission and
# CVLLM=128 controls request concurrency.
set -uo pipefail
cd "$(dirname "$0")/../.."
unset http_proxy https_proxy; export no_proxy='localhost,127.0.0.1'
export TOKENIZER_PATH=${TOKENIZER_PATH:-/nas/lzc/model/qwen3-4b-instruct-2507}
PY=${PY:-/usr/bin/python3}
OUT=${OUT:-results/eval}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
GROUP_CAP=${GROUP_CAP:-10}
DHIGH=${DHIGH:-512}
CVLLM=${CVLLM:-128}
: "${SERVERS:?set SERVERS}"; : "${JUDGE_URL:?set JUDGE_URL}"

DATASETS=${DATASETS:-$($PY -c 'from eval.matrix import SINGLE_CORE;print(",".join(SINGLE_CORE))')}
$PY -m eval.pipeline --model recurrent --servers "$SERVERS" \
  --judge-url "$JUDGE_URL" --out-dir "$OUT" \
  --group-cap "$GROUP_CAP" --dhigh-vllm "$DHIGH" --c-vllm "$CVLLM" $LIMIT_ARG \
  --datasets "$DATASETS"
