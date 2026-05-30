#!/usr/bin/env bash
# NJU Rule RAG — one-click production start.
#
# Usage:
#   ./scripts/start_server.sh               # production (port 8000)
#   ./scripts/start_server.sh --reload      # dev mode with auto-reload
#   ./scripts/start_server.sh --port 9000   # custom port
#
# This script:
#   1. Preserves proxy env vars for HuggingFace access (WSL inherits from Windows)
#   2. Excludes local services (Ollama) from proxy
#   3. Enables PyTorch expandable_segments (GPU memory fragmentation fix)
#   4. Activates the Python venv
#   5. Runs preflight checks
#   6. Starts uvicorn

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== NJU Rule RAG — starting server ==="

# ── 1. Proxy ────────────────────────────────────────────────────
# WSL2 inherits Windows proxy settings.  HuggingFace model checks need
# external access, so we keep HTTP_PROXY / HTTPS_PROXY if present.
# Local services (Ollama) MUST bypass the proxy.
echo "[env] Proxy: HTTP_PROXY=${HTTP_PROXY:-unset} HTTPS_PROXY=${HTTPS_PROXY:-unset}"
export NO_PROXY="localhost,127.0.0.1,.local,ollama,host.docker.internal${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

# ── 2. HuggingFace — use network (proxy handles external access) ─
# No longer forcing HF_HUB_OFFLINE=1; weights should already be cached.
# If a model needs to download fresh files, the proxy makes it work.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE

# ── 3. PyTorch GPU memory optimisation ────────────────────────────
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
