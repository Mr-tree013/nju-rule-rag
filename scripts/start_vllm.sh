#!/bin/bash
# Start vLLM server with LoRA adapter (ROADMAP_LORA §5)
# Usage: bash scripts/start_vllm.sh [base|nju-lora]

MODEL=${1:-nju-lora}
BASE_MODEL="Qwen/Qwen3-8B"
LORA_PATH="data/lora_adapters/nju-v1"
PORT=8001

echo "=== Starting vLLM ($MODEL) ==="

if [ "$MODEL" = "base" ]; then
    echo "Mode: base model only (no LoRA)"
    vllm serve "$BASE_MODEL" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --port $PORT \
        --dtype bfloat16
elif [ "$MODEL" = "nju-lora" ]; then
    if [ ! -d "$LORA_PATH" ]; then
        echo "ERROR: LoRA adapter not found at $LORA_PATH"
        echo "Run: python scripts/lora_train.py first"
        exit 1
    fi
    echo "Mode: base + nju-lora adapter"
    vllm serve "$BASE_MODEL" \
        --enable-lora \
        --lora-modules nju-lora="$LORA_PATH" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --port $PORT \
        --dtype bfloat16
else
    echo "Unknown model: $MODEL (use 'base' or 'nju-lora')"
    exit 1
fi
