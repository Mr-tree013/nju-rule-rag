"""
Deep health check aggregating Ollama, GPU, model, index, and cache status.
"""

import time
import json
import urllib.request
from pathlib import Path
from typing import Any


def get_deep_health(project_root: Path, cache_stats_fn=None) -> dict[str, Any]:
    """Return a comprehensive runtime health snapshot."""

    health: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── Ollama ───────────────────────────────────────────────────
    health["ollama"] = _check_ollama()

    # ── GPU ──────────────────────────────────────────────────────
    health["gpu"] = _check_gpu()

    # ── Models loaded ────────────────────────────────────────────
    health["models_loaded"] = _check_models_loaded()

    # ── Index ────────────────────────────────────────────────────
    health["index"] = _check_index(project_root)

    # ── Cache ────────────────────────────────────────────────────
    if cache_stats_fn:
        try:
            health["cache"] = cache_stats_fn()
        except Exception:
            health["cache"] = {"error": "无法获取缓存状态"}
    else:
        health["cache"] = {"hits": 0, "misses": 0, "size": 0}

    return health


def _check_ollama() -> dict[str, Any]:
    result: dict[str, Any] = {"reachable": False, "models": [], "latency_ms": 0}
    t0 = time.time()
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        result["reachable"] = True
        result["latency_ms"] = round((time.time() - t0) * 1000)
        result["models"] = [m["name"] for m in data.get("models", [])]
        result["models_detail"] = [
            {"name": m["name"], "size_mb": round(m.get("size", 0) / 1e6)}
            for m in data.get("models", [])
        ]
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


def _check_gpu() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            result = {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "total_mb": round(total_bytes / (1024 * 1024)),
                "used_mb": round((total_bytes - free_bytes) / (1024 * 1024)),
                "free_mb": round(free_bytes / (1024 * 1024)),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass
    return result


def _check_models_loaded() -> dict[str, bool]:
    result = {"bge_m3": False, "bge_reranker": False}
    try:
        from app.pipeline import _pipeline
        if _pipeline is not None:
            # Check if embedding model is loaded
            r = _pipeline._retriever
            if hasattr(r, "_vector") and r._vector is not None:
                result["bge_m3"] = r._vector.embedding_model is not None
            # Check if reranker is loaded
            reranker = getattr(_pipeline, "_reranker", None)
            if reranker is not None:
                result["bge_reranker"] = reranker.is_loaded
    except Exception:
        pass
    return result


def _check_index(project_root: Path) -> dict[str, Any]:
    index_dir = project_root / "data" / "index"
    chunks_file = project_root / "data" / "chunks" / "chunks.jsonl"
    chunks_count = 0
    try:
        import json
        with open(chunks_file, encoding="utf-8") as f:
            chunks_count = sum(1 for _ in f)
    except Exception:
        pass

    return {
        "chunks": chunks_count,
        "bm25_loaded": (index_dir / "bm25.pkl").exists(),
        "vector_loaded": (index_dir / "chroma" / "chroma.sqlite3").exists(),
    }
