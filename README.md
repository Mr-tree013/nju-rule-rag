# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 70 份校规、办事指南和校园生活文档，支持自然语言提问、来源引用、风险分级与拒答机制。已接入 QQ Bot（NapCat + OneBot v11）。

---

## 运行环境

本项目已全面迁移到**本地模型**，需要 NVIDIA GPU 运行。

| 组件 | 技术栈 | 显存占用 |
|------|--------|---------|
| LLM 生成 | Ollama + Qwen3-8B（无思考模式） | ~5 GB |
| Embedding | BGE-M3（1024 维，sentence-transformers） | ~2.2 GB |
| Reranker | BGE-Reranker-v2-m3（cross-encoder） | ~1 GB |
| **合计** | | **~8-10 GB** |

### 最低硬件要求

- **GPU**: NVIDIA RTX 3060 12GB 或更高（推荐 RTX 4070 Ti Super 16GB）
- **显存**: ≥12 GB
- **内存**: ≥16 GB 系统 RAM
- **磁盘**: ≥30 GB（模型文件约 8 GB + 索引约 2 GB）
- **OS**: Linux（推荐 WSL2 + Windows 11）

### 开发环境

本项目运行在 **Windows 11 + WSL2（Ubuntu）**：

- GPU: NVIDIA GeForce RTX 4070 Ti Super (16 GB VRAM)
- Driver: 551.52 (CUDA 12.4)
- PyTorch: 2.6.0+cu124（必须与驱动 CUDA 版本匹配）
- Python: 3.12

#### WSL2 网络配置

在 Windows 用户目录创建 `%USERPROFILE%\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
```

执行 `wsl --shutdown` 重启 WSL。mirrored 模式让 WSL 与 Windows 共享 IP，NapCat 可通过 `127.0.0.1` 访问 WSL 服务。

---

## 快速部署

```bash
# 1. 安装 Ollama 并拉取模型
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink

# 2. 克隆项目并安装依赖
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
python -m venv .venv
source .venv/bin/activate

# 3. PyTorch GPU 版（版本必须匹配 nvidia-smi 显示的 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 配置
cp .env.example .env   # 编辑填入 API key（Ollama 用 ollama 占位即可）

# 6. 构建索引（首次需下载 BGE-M3 ~2.2GB）
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_chunks.py

# 7. 启动（预热查询会提前加载所有模型到 GPU，约 15 秒）
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

看到 `[Pipeline] 预热完成` 后即可使用。后续请求延迟 ~2 秒。

---

## QQ Bot 接入

基于 NapCat（OneBot v11 HTTP 回调）。

### 配置

`.env` 中设置：

```bash
QQ_BOT_SELF_ID=你的机器人QQ号
QQ_BOT_API_BASE_URL=http://127.0.0.1:8000
```

NapCat WebUI → 网络配置 → 新建 HTTP 客户端：

| 字段 | 值 |
|------|-----|
| 名称 | `nju-rule-rag` |
| URL | `http://127.0.0.1:8000/qq` |
| 消息格式 | `string` |
| 启用 | ✅ |

群内发送 `/问 你的问题` 或 `/ask 你的问题` 即可。

---

## 在线端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | 问答 `{"question": "..."}` |
| `/ask/stream` | POST | SSE 流式问答 |
| `/feedback` | POST | 用户反馈 `{"question","rating":"up"/"down"}` |
| `/cache/stats` | GET | 缓存命中统计 |
| `/qq` | POST | QQ Bot webhook |

### /ask 响应

```json
{
  "question": "补考没过怎么办？",
  "answer": "补考没过需重修...",
  "risk_level": "medium",
  "need_human_confirm": true,
  "sources": [{"chunk_id", "source_id", "title", "url", "priority", "fetched_at"}],
  "debug": {"retrieval_count": 40, "latency": 1.95, "llm_used": "qwen3:8b-nothink", "cached": false}
}
```

---

## 架构

```
POST /ask {"question": "..."}
        │
        ▼
[_handle_meta_question]  "你是谁"/"你能干什么" → 直接回复（不走检索）
        │
        ▼
[QueryRewriter]          口语化规范化（should_rewrite()守卫）
        │
        ▼
TwoLayerRiskClassifier   L1关键词(高召回) → L2 BGE-M3 centroid消歧
        │
        ▼
HybridRetriever          BM25(0.25) + BGE-M3 Vector(0.45) + Priority(0.30)
        │
        ▼
CrossEncoderReranker     BGE-Reranker-v2-m3 (40候选 → 12精排)
        │
        ▼
_filter → _dedup_chunks  (3/source, 12 total)
        │
        ▼
LLM (Qwen3-8B)           fallback→DeepSeek on failure
        │
        ▼
[_verify_citations]      bigram重叠度校验（可选）
        │
        ▼
_format_response         来源去重 + 高风险通知(含部门联系方式)
        │
        ▼
{ question, answer, risk_level, sources[], debug }
```

### 服务模块

| 文件 | 职责 |
|------|------|
| `app/main.py` | FastAPI 入口，所有端点 + CORS |
| `app/pipeline.py` | RAGPipeline，每步骤可覆写（依赖注入），含预热 |
| `app/config.py` | Frozen Settings dataclass，`.env` 驱动 |
| `app/retriever.py` | BM25 + Chroma 混合检索，协议接口 |
| `app/reranker.py` | CrossEncoderReranker（BGE-Reranker-v2-m3） |
| `app/query_rewriter.py` | 口语查询改写（触发守卫：≤6字 或 口语词） |
| `app/llm_client.py` | OpenAI 兼容客户端（chat + stream + 3次退避重试） |
| `app/policy.py` | TwoLayerRiskClassifier + ResponseTemplates |
| `app/cache.py` | LRU 内存缓存（200条, 1h TTL） |
| `app/qq_bot.py` | QQ Bot 适配器 |

### 评测脚本

| 脚本 | 用途 |
|------|------|
| `eval_rag.py` | 70 题端到端 /ask 评测（需服务器） |
| `eval_retrieval.py` | 检索指标（recall@k, MRR, precision/recall），支持 `--rerank` |
| `eval_generation.py` | LLM-as-judge 评分（忠实度/相关性/拒答） |
| `tune_weights.py` | 混合权重网格搜索（126 组合） |
| `check_regression.py` | CI 门禁（对比基线，回退则非零退出） |
| `annotate_gold_sources.py` | 标注评测题金标来源 |

---

## 数据流水线

```
sources.csv (70) → processed/*.md → build_chunks.py → chunks.jsonl (3771)
                                                              │
                                    build_index.py ──────────┘
                                    BM25 (jieba) + Chroma (BGE-M3, 1024-dim)
```

文档更新后重建索引：

```bash
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py
```

---

## 模型清单

| 模型 | 大小 | 用途 |
|------|------|------|
| Qwen3-8B (Ollama) | 5.2 GB | LLM 生成 |
| BGE-M3 | 2.2 GB | 文本向量化（1024 维） |
| BGE-Reranker-v2-m3 | 1.0 GB | 检索精排 |

---

## 检索基线

70 题评测（BGE-M3 + Reranker + 优化权重 + 改进 Chunking）：

| 指标 | 值 |
|------|-----|
| recall@5 | 84.3% |
| recall@10 | 92.9% |
| MRR | 0.599 |
| 测试 | 120 个单元测试 |

---

## 配置参考

```bash
# LLM（本地）
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b-nothink

# LLM 回退
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_API_KEY=sk-your-key
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# Embedding
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3

# Reranker
ENABLE_RERANK=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_K=40
RERANK_TOP_K=12

# 查询改写
ENABLE_QUERY_REWRITE=true

# 检索参数（网格搜索优化）
BM25_TOP_K=10; VECTOR_TOP_K=10; HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2; HIGH_RISK_MIN_SCORE=0.25

# QQ Bot
QQ_BOT_SELF_ID=你的QQ号
QQ_BOT_API_BASE_URL=http://127.0.0.1:8000
```

---

## 常见问题

### 启动报 CUDA 不可用

PyTorch 版本需与驱动 CUDA 版本一致：

```bash
python -c "import torch; print(torch.version.cuda)"  # 应显示 12.4
nvidia-smi | grep "CUDA Version"                       # 应一致
```

不一致则重装：`pip install torch --index-url https://download.pytorch.org/whl/cu124`

### 显存不足

```bash
ENABLE_RERANK=false        # 关 reranker（省 ~1 GB）
ENABLE_VECTOR=false        # 关向量检索，纯 BM25（省 ~2.2 GB）
# 换成云端 API 替代本地 LLM 可省 ~5 GB
```

### 构建命令报 "No module named 'app'"

需要 `PYTHONPATH=.`：

```bash
PYTHONPATH=. python scripts/build_index.py
```

### QQ Bot 首条消息很慢

服务启动时会自动跑预热查询（看到 `[Pipeline] 预热完成` 即完成），之后请求 ~2 秒。

---

> 本系统仅提供一般性校规查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
