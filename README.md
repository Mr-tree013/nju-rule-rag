# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 70 份校规、办事指南和校园生活文档，支持自然语言提问、来源引用、风险分级与拒答机制。已接入 QQ Bot（NapCat + OneBot v11）。

---

## 运行环境

本项目在当前版本（v0.4）已全面迁移到**本地模型**，需要 NVIDIA GPU 运行。

| 组件 | 技术栈 | 显存占用 |
|------|--------|---------|
| LLM 生成 | Ollama + Qwen3-8B（无思考模式） | ~5 GB |
| Embedding | BGE-M3（1024 维，sentence-transformers） | ~2.2 GB |
| Reranker | BGE-Reranker-v2-m3（cross-encoder） | ~1 GB |
| **合计** | | **~8-10 GB** |

### 最低硬件要求

- **GPU**: NVIDIA RTX 3060 12GB 或更高（推荐 RTX 4070 Ti Super 16GB）
- **显存**: ≥12 GB（同时加载 LLM + Embedding + Reranker 约需 10 GB）
- **内存**: ≥16 GB 系统 RAM
- **磁盘**: ≥30 GB（模型文件约 8 GB + 项目约 2 GB）
- **OS**: Linux（推荐 WSL2 + Windows 11，见下文）

### 开发环境说明

本项目开发环境为 **Windows 11 + WSL2（Ubuntu）**，配置如下：

- GPU: NVIDIA GeForce RTX 4070 Ti Super (16 GB VRAM)
- Driver: 551.52 (CUDA 12.4)
- WSL2 网络模式: `mirrored`（使局域网设备可访问 WSL 服务）
- PyTorch: 2.6.0+cu124（必须与驱动 CUDA 版本匹配）
- Python: 3.12

#### WSL2 配置（Windows 用户必读）

在 Windows 用户目录下创建 `%USERPROFILE%\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
```

然后 PowerShell 执行 `wsl --shutdown` 重启 WSL。

> `mirrored` 模式让 WSL 与 Windows 共享 IP，手机/其他设备可直接访问 WSL 内的服务（如 QQ Bot webhook）。

---

## 快速部署

### 1. 环境准备

```bash
# 安装 Ollama（Linux / WSL2）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen3:8b
# 创建无思考模式变体（避免输出 <think> 块浪费 token）
ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink

# 克隆项目
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
python -m venv .venv
source .venv/bin/activate
```

### 2. 安装 PyTorch（GPU 版）

**必须安装与 NVIDIA 驱动 CUDA 版本匹配的 PyTorch**。查看驱动 CUDA 版本：

```bash
nvidia-smi  # 右上角显示 CUDA Version
```

```bash
# CUDA 12.4 驱动（本项目环境）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 其他版本请参考 https://pytorch.org/get-started/locally/
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置

```bash
cp .env.example .env
```

编辑 `.env`，关键配置：

```bash
# LLM — 本地 Qwen3-8B（Ollama OpenAI 兼容端点）
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b-nothink

# LLM 回退 — 本地模型故障时切到 DeepSeek
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_API_KEY=sk-your-deepseek-key
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# Embedding — BGE-M3（1024 维，中文语义检索）
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3

# Reranker — 二阶段精排
ENABLE_RERANK=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 5. 构建索引

```bash
# 重建 chunks（3771 个片段）
PYTHONPATH=. python scripts/build_chunks.py

# 构建 BM25 + Chroma 向量索引（首次需下载 BGE-M3 ~2.2GB）
PYTHONPATH=. python scripts/build_index.py

# 验证
python scripts/validate_chunks.py
```

### 6. 启动

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

首次请求会触发模型加载（约 10-15 秒），后续请求延迟 ~2 秒。

---

## 模型清单

首次部署需要下载以下模型文件（合计约 8 GB），之后缓存在本地：

| 模型 | 大小 | 用途 | 下载方式 |
|------|------|------|---------|
| Qwen3-8B (ollama) | 5.2 GB | LLM 生成回答 | `ollama pull qwen3:8b` |
| BGE-M3 | 2.2 GB | 文本向量化 | 首次 `build_index.py` 自动下载 |
| BGE-Reranker-v2-m3 | 1.0 GB | 检索精排 | 首次 `/ask` 请求自动下载 |
| text2vec-base-chinese | 0.4 GB | （旧版，已废弃） | 不再使用 |

> 所有 HuggingFace 模型首次下载后缓存于 `~/.cache/huggingface/`，后续启动无需联网。

---

## QQ Bot 接入

基于 NapCat（OneBot v11 HTTP 回调）实现 QQ 群问答。

### 1. 配置 .env

```bash
QQ_BOT_SELF_ID=你的机器人QQ号
QQ_BOT_API_BASE_URL=http://127.0.0.1:8000
```

### 2. 配置 NapCat

登录 NapCat WebUI → 网络配置 → 新建 HTTP 客户端：

| 字段 | 值 |
|------|-----|
| 名称 | `nju-rule-rag` |
| URL | `http://127.0.0.1:8000/qq` |
| 消息格式 | `string` |
| 启用 | ✅ |

### 3. 使用

群内发送 `/问 你的问题` 或 `/ask 你的问题` 即可。

---

## 数据流水线

```
sources.csv (70条来源) → processed/*.md → build_chunks.py → chunks.jsonl (3771条)
                                                                    │
                                          build_index.py ────────────┘
                                          BM25 (jieba) + Chroma (BGE-M3, 1024-dim)
```

### 更新索引（添加新文档后）

```bash
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py
```

### 文档格式转换

```bash
python scripts/parse_to_markdown.py input.html --title "标题" -o data/processed/输出.md
python scripts/parse_to_markdown.py input.pdf  --title "标题" -o data/processed/输出.md
```

---

## 架构

```
POST /ask {"question": "..."}
        │
        ▼
[Query Rewrite]  口语化问题规范化（可选）
        │
        ▼
RiskClassifier  关键词风险分级（low/medium/high）
        │
        ▼
HybridRetriever BM25(0.25) + Vector BGE-M3(0.45) + Priority(0.30)
        │
        ▼
CrossEncoderReranker  BGE-Reranker-v2-m3 二阶段精排
        │
        ▼
_filter → _dedup_chunks  (3/source, 12 total)
        │
        ▼
LLM (Qwen3-8B / fallback DeepSeek)  有依据归纳 + 格式输出
        │
        ▼
{ question, answer, risk_level, sources[], debug }
```

### 服务模块

- **`app/main.py`** — FastAPI 入口。`GET /health`，`POST /ask`，`POST /qq`
- **`app/config.py`** — 不可变 Settings dataclass，`.env` 驱动
- **`app/pipeline.py`** — RAGPipeline，每步骤可覆写（依赖注入）
- **`app/retriever.py`** — BM25 + Chroma 向量混合检索器
- **`app/reranker.py`** — Cross-encoder 二阶段精排
- **`app/query_rewriter.py`** — 口语化查询改写（触发守卫）
- **`app/llm_client.py`** — OpenAI 兼容 LLM 客户端（3 次退避重试）
- **`app/policy.py`** — 风险分类 + 回复模板
- **`app/qq_bot.py`** — QQ Bot 适配器（OneBot v11）

### 评测体系

| 脚本 | 用途 |
|------|------|
| `scripts/eval_rag.py` | 70 题端到端 /ask 评测 |
| `scripts/eval_retrieval.py` | 检索层指标（recall@k, MRR, precision/recall） |
| `scripts/eval_generation.py` | 生成层 LLM-as-judge 评分（忠实度/相关性/拒答） |
| `scripts/tune_weights.py` | 混合权重网格搜索 |
| `scripts/check_regression.py` | CI 回归门禁（对比基线，质量回退则非零退出） |
| `scripts/annotate_gold_sources.py` | 给评测题标注金标来源 |

---

## 在线端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查（含 retriever 状态） |
| `/ask` | POST | 问答接口 `{"question": "..."}` |
| `/qq` | POST | QQ Bot webhook（NapCat OneBot v11 回调） |

### /ask 响应格式

```json
{
  "question": "补考没过怎么办？",
  "answer": "补考没过需重修...",
  "risk_level": "medium",
  "need_human_confirm": true,
  "sources": [{"chunk_id", "source_id", "title", "url", "priority"}],
  "debug": {"retrieval_count": 40, "latency": 1.95, "llm_used": "qwen3:8b-nothink"}
}
```

---

## 评测基线

70 题检索评测（BGE-M3 + Reranker + 优化权重 + 改进 Chunking）：

| 指标 | 值 |
|------|-----|
| recall@5 | 84.3% |
| recall@10 | 92.9% |
| MRR | 0.599 |
| 端到端成功 | 70/70 |

---

## 配置参考

```bash
# ── LLM（本地 Qwen3-8B）──
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b-nothink

# ── LLM 回退（可选）──
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_API_KEY=sk-your-deepseek-key
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# ── Embedding ──
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3

# ── Reranker ──
ENABLE_RERANK=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_K=40
RERANK_TOP_K=12

# ── 查询改写 ──
ENABLE_QUERY_REWRITE=true

# ── 检索调参 ──
BM25_TOP_K=10
VECTOR_TOP_K=10
HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2
HIGH_RISK_MIN_SCORE=0.25

# ── 向量检索 ──
ENABLE_VECTOR=true

# ── QQ Bot ──
QQ_BOT_SELF_ID=你的QQ号
QQ_BOT_API_BASE_URL=http://127.0.0.1:8000
```

---

## 资料库概况

| 类别 | 数量 |
|------|------|
| 来源总数 | 70 |
| 本科生院文档 | 47 |
| 南哪助手生活指南 | 22 |
| 可检索片段 | 3,771 |
| 评测问题 | 70 题（标注金标来源） |
| 测试覆盖 | 110 个单元测试 |

---

## 常见问题

### Q: 启动时报 CUDA 不可用？

检查 PyTorch 版本是否与驱动 CUDA 匹配：

```bash
python -c "import torch; print(torch.version.cuda)"  # 应显示 12.4
nvidia-smi | grep "CUDA Version"                       # 应一致
```

若不一致，重新安装对应版本的 PyTorch。

### Q: 显存不足？

按需关闭组件：

```bash
ENABLE_RERANK=false        # 关闭 reranker（省 ~1 GB）
ENABLE_VECTOR=false        # 关闭向量检索（纯 BM25，省 ~2.2 GB）
# 使用云端 API 替代本地 LLM 可省 ~5 GB
```

### Q: BM25 索引报错 "No module named 'app'"？

构建命令需加 `PYTHONPATH=.`：

```bash
PYTHONPATH=. python scripts/build_index.py
```

---

> 本系统仅提供一般性校规和生活信息查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
