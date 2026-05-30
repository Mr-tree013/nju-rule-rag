#!/usr/bin/env bash
# Start Ollama with Flash Attention + KV cache quantization for better GPU memory.
#
# Usage:
#   ./scripts/ollama_env.sh    # prints the command to restart Ollama
#   source scripts/ollama_env.sh && ollama serve  # apply and start
#
# Environment variables (must be set BEFORE ollama serve):
#   OLLAMA_FLASH_ATTENTION=1   — enable Flash Attention (Ada arch, CC 8.9)
#   OLLAMA_KV_CACHE_TYPE=q8_0  — 8-bit KV cache, ~50% memory savings vs fp16
#   OLLAMA_KEEP_ALIVE=24h      — keep model loaded, avoid cold starts

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_KEEP_ALIVE=24h

echo "[Ollama] Flash Attention: ${OLLAMA_FLASH_ATTENTION:-unset}"
echo "[Ollama] KV Cache type:   ${OLLAMA_KV_CACHE_TYPE:-unset}"
echo "[Ollama] Keep Alive:       ${OLLAMA_KEEP_ALIVE:-unset}"
echo ""
echo "Now run: ollama serve"
