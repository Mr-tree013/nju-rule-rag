#!/usr/bin/env bash
# NJU Rule RAG — one-click production start.
#
# Usage:
#   ./scripts/start_server.sh               # production (port 8000)
#   ./scripts/start_server.sh --reload      # dev mode with auto-reload
#   ./scripts/start_server.sh --port 9000   # custom port
#
# This script replaces the manual startup checklist.  It:
#   1. Clears stale proxy env vars (WSL → Windows proxy leak)
#   2. Sets HuggingFace offline mode
#   3. Enables PyTorch expandable_segments (GPU memory fragmentation fix)
#   4. Activates the Python venv
#   5. Runs preflight checks
#   6. Starts uvicorn

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NJU Rule RAG — starting server ==="

# ── 1. Clear stale proxy vars ─────────────────────────────────────
# Windows system proxy leaks into WSL2 and breaks HuggingFace requests.
echo "[env] Clearing proxy variables..."
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy NO_PROXY no_proxy
# also unset uppercase variants that some tools read
unset HTTP_PROXY HTTPS_PROXY HTTP_PROXY http_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

# ── 2. HuggingFace offline ────────────────────────────────────────
# Prevent HF Hub HEAD requests that hang on dead proxies.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── 3. PyTorch GPU memory optimisation ────────────────────────────
# expandable_segments reduces CUDA memory fragmentation — critical for
# long-running servers that hold 3 models in 16GB VRAM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── 4. Activate venv ──────────────────────────────────────────────
if [ -d ".venv" ]; then
    echo "[venv] Activating..."
    source .venv/bin/activate
else
    echo "[venv] WARNING: .venv not found, using system python"
fi

# ── 5. Preflight checks ───────────────────────────────────────────
if [ -f "scripts/preflight_check.py" ]; then
    echo "[preflight] Running preflight checks..."
    python scripts/preflight_check.py || {
        echo "[preflight] WARNING: checks failed — starting anyway in 3s..."
        sleep 3
    }
else
    echo "[preflight] Skipped (scripts/preflight_check.py not found)"
fi

# ── 6. Start server ───────────────────────────────────────────────
echo ""
echo "=== Starting uvicorn on http://0.0.0.0:8000 ==="
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
