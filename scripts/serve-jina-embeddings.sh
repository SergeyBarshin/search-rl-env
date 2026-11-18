#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
local_server=$root/data/llama.cpp/b12500/llama-b12500/llama-server
if [ -x "$local_server" ]; then
    default_server=$local_server
else
    default_server=llama-server
fi
model=${JINA_GGUF_PATH:-"$root/models/jina-embeddings-v5-text-small/v5-small-retrieval-Q4_K_M.gguf"}
expected=9440cf89f3e8a7a31a42e11b87e106dd5b344af4e0e3b6b21a96136cc8686e21
printf '%s  %s\n' "$expected" "$model" | shasum -a 256 -c -
exec "${LLAMA_SERVER:-$default_server}" --model "$model" --alias jina-v5-small-retrieval-q4_k_m \
    --host 127.0.0.1 --port "${JINA_PORT:-8081}" --embeddings --pooling last \
    --n-gpu-layers 99 --ctx-size 8192 --batch-size 2048 --ubatch-size 2048 --parallel 1
