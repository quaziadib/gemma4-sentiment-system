#!/usr/bin/env bash
# Starts llama-cpp-python OpenAI-compatible server for Gemma 4 26B A4B GGUF.
# VLLM 0.19.0 does not support the gemma4 GGUF architecture; llama.cpp does.
set -euo pipefail

# Path to the downloaded GGUF file.
# huggingface-cli download puts it here by default:
#   ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/
# Run: huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
#        gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
# then set GGUF_PATH to the printed local path, or set HF_HUB_CACHE below.
GGUF_PATH="${GGUF_PATH:-}"

if [[ -z "${GGUF_PATH}" ]]; then
    echo "Resolving GGUF path from HF cache..."
    GGUF_PATH=$(python3 - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
    filename="gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
)
print(path)
EOF
)
    echo "  -> ${GGUF_PATH}"
fi

# Tune these to your hardware.
# -1 = offload all layers to GPU (RTX 4090, 24 GB VRAM).
# Reduce N_GPU_LAYERS (e.g. 40) if you hit OOM.
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
CTX_SIZE="${CTX_SIZE:-4096}"
HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${SERVER_PORT:-8000}"

echo "Starting llama-cpp-python server"
echo "  Model       : ${GGUF_PATH}"
echo "  GPU layers  : ${N_GPU_LAYERS}"
echo "  Context     : ${CTX_SIZE}"
echo "  Listen      : ${HOST}:${PORT}"

python3 -m llama_cpp.server \
    --model "${GGUF_PATH}" \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${CTX_SIZE}" \
    --n_batch 512 \
    --host "${HOST}" \
    --port "${PORT}" \
    --chat_format gemma
