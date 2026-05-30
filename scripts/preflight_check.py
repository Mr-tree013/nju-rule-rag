#!/usr/bin/env python3
"""Preflight checks for NJU Rule RAG server startup.

Run before starting the server to catch common failures early:
  python scripts/preflight_check.py

Exit code 0 = all checks passed. Non-zero = fix the reported issues first.

Checks:
  1. CUDA available + driver/PyTorch version match
  2. Embedding / reranker model weights present
  3. Ollama reachable, qwen3:8b-nothink available
  4. GPU VRAM >= 4 GB free
  5. No stale proxy environment variables
  6. BM25 and Chroma index present
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXIT = 0


def ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")


def warn(msg: str) -> None:
    global EXIT
    print(f"  \033[33m⚠\033[0m {msg}")
    if EXIT == 0:
        EXIT = 1


def fail(msg: str) -> None:
    global EXIT
    print(f"  \033[31m✗\033[0m {msg}")
    EXIT = 2


def header(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ── 1. CUDA ──────────────────────────────────────────────────────

header("1. CUDA")
try:
    import torch
    if torch.cuda.is_available():
        cuda_ver = torch.version.cuda
        driver_ver = torch.cuda.get_device_properties(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        ok(f"CUDA {cuda_ver} — {torch.cuda.get_device_name(0)} ({mem_total:.1f} GB)")
        # Check driver/PyTorch version consistency
        if cuda_ver:
            try:
                driver_major = int(str(cuda_ver).split(".")[0])
                runtime_ver = torch._C._cuda_getDriverVersion()
                runtime_major = runtime_ver // 1000
                if driver_major != runtime_major:
                    warn(f"CUDA 版本不一致: PyTorch 编译={cuda_ver}, 驱动={runtime_major}.x")
            except Exception:
                pass
    else:
        fail("CUDA 不可用 — 推理性能会显著下降")
except ImportError as e:
    fail(f"无法导入 torch: {e}")
except Exception as e:
    fail(f"CUDA 检查异常: {e}")

# ── 2. Model weights ─────────────────────────────────────────────

header("2. Model weights")
hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
hf_hub = os.path.expanduser(os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub")))

model_dirs = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
}
for name, model_id in model_dirs.items():
    model_path = Path(hf_hub) / ("models--" + model_id.replace("/", "--"))
    if model_path.exists() and list(model_path.glob("snapshots/*")):
        ok(f"{name} ({model_id})")
    else:
        warn(f"{name} ({model_id}) 未找到 — 首次加载时会从 HuggingFace 下载（需要网络）")

# ── 3. Ollama ────────────────────────────────────────────────────

header("3. Ollama")
ollama_url = "http://localhost:11434"
try:
    req = urllib.request.Request(f"{ollama_url}/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    models = [m["name"] for m in data.get("models", [])]
    ok(f"Ollama 可达 — {len(models)} 个模型")
    if "qwen3:8b-nothink" in models:
        ok("qwen3:8b-nothink 已安装")
    else:
        fail(
            "qwen3:8b-nothink 未找到！\n"
            "    请运行: ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink"
        )
except urllib.error.URLError as e:
    fail(f"Ollama 不可达 ({ollama_url}): {e}\n"
         "    请确认 Ollama 已启动")
except Exception as e:
    fail(f"Ollama 检查失败: {e}")

# ── 4. GPU VRAM ─────────────────────────────────────────────────

header("4. GPU 显存")
try:
    import torch
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        used_gb = total_gb - free_gb
        if free_gb >= 4:
            ok(f"空闲 {free_gb:.1f} GB / 总计 {total_gb:.1f} GB")
        else:
            warn(
                f"显存紧张！空闲 {free_gb:.1f} GB / 总计 {total_gb:.1f} GB\n"
                f"    建议: 关闭其他 GPU 程序, 或设置 RERANKER_DEVICE=cpu"
            )
except Exception:
    warn("无法查询显存状态")

# ── 5. Proxy environment ────────────────────────────────────────

header("5. 代理变量")
proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
active_proxy = {k: v for k, v in os.environ.items() if k in proxy_vars and v}
no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""

if active_proxy:
    has_localhost = "localhost" in no_proxy or "127.0.0.1" in no_proxy
    if has_localhost:
        ok(f"代理已配置: {', '.join(f'{k}={v[:30]}...' for k, v in active_proxy.items())}, NO_PROXY 已排除本地服务")
    else:
        warn(
            f"代理已配置但 NO_PROXY 未包含 localhost，可能影响 Ollama 连接\n"
            "    建议: export NO_PROXY=localhost,127.0.0.1"
        )
else:
    warn("未检测到代理 — HuggingFace 下载/检查可能失败（如仅用缓存则无影响）")

# ── 6. Index files ──────────────────────────────────────────────

header("6. 索引文件")
index_dir = PROJECT_ROOT / "data" / "index"
chunks_file = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

if chunks_file.exists():
    ok(f"chunks 文件: {chunks_file}")
else:
    fail(f"chunks 文件不存在: {chunks_file}\n"
         "    请运行: PYTHONPATH=. python scripts/build_chunks.py")

checks = [
    ("bm25.pkl", "BM25 索引"),
    ("chunk_lookup.json", "chunk 查找表"),
]
has_vec = False
for fname, label in checks:
    fpath = index_dir / fname
    if fpath.exists():
        ok(f"{label}: {fname}")
    else:
        fail(f"{label} ({fname}) 不存在")

chroma_dir = index_dir / "chroma"
if chroma_dir.exists() and (chroma_dir / "chroma.sqlite3").exists():
    ok(f"Chroma 向量索引: chroma/")
    has_vec = True
else:
    warn(f"Chroma 向量索引不完整\n"
         "    请运行: PYTHONPATH=. python scripts/build_index.py")

# ── Summary ──────────────────────────────────────────────────────

print()
if EXIT == 0:
    print("\033[32m\033[1m所有检查通过 ✓\033[0m")
    print("运行 ./scripts/start_server.sh 启动服务")
elif EXIT == 1:
    print("\033[33m\033[1m有警告，服务仍可启动\033[0m")
else:
    print("\033[31m\033[1m有错误，请先修复后再启动！\033[0m")

sys.exit(EXIT)
