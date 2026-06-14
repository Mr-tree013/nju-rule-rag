# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 220 份校规、办事指南和校园生活文档（3,441 chunks），支持自然语言提问、来源引用、风险分级与拒答机制。已接入 QQ Bot（NapCat + OneBot v11）。

**当前版本**: v0.7.0 | 145 道评测题 | 延迟 1.76s | 忠实度 3.30/5 | 来源覆盖率 100% | 身份: 南鉴Bot

---

## 快速开始

```bash
# 1. 安装 Ollama 并创建模型
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b
ollama create qwen3:8b-nothink-v2 -f scripts/modelfile.qwen3-nothink-v2

# 2. 克隆项目
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag

# 3. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 4. 安装依赖（PyTorch GPU 版本需匹配 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 5. 配置环境变量
cp .env.example .env   # 编辑填入 API key（Ollama 用 ollama 占位即可）

# 6. 构建索引（首次需下载 BGE-M3 ~2.2GB）
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py

# 7. 一键启动
./scripts/start_server.sh           # 生产模式
./scripts/start_server.sh --reload  # 开发模式（自动重载）
./scripts/start_daemon.sh           # 后台守护（tmux/nohup 自动检测）
```

看到 `[Pipeline] 预热完成` 后即可使用。

---

## 部署环境

### 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 12GB | RTX 4070 Ti Super 16GB |
| 显存 | ≥12 GB | ≥16 GB |
| 内存 | ≥16 GB | ≥32 GB |
| 磁盘 | ≥30 GB | ≥50 GB |

### 软件依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.12+ | |
| PyTorch | 2.6.0+cu124 | 必须与 nvidia-smi 的 CUDA 版本一致 |
| Ollama | 0.30+ | 提供 Qwen3-8B 推理服务 |
| CUDA Driver | ≥12.4 | |

### WSL2 配置（Windows 用户）

在 `%USERPROFILE%\.wslconfig` 中：

```ini
[wsl2]
networkingMode=mirrored
```

执行 `wsl --shutdown` 重启 WSL。

### Ollama 环境变量（必须在 ollama serve 的环境中设置）

```bash
export OLLAMA_FLASH_ATTENTION=1     # Flash Attention 加速
export OLLAMA_KV_CACHE_TYPE=q8_0    # 8-bit KV cache（省 ~50% 显存）
export OLLAMA_KEEP_ALIVE=24h        # 保持模型常驻内存
```

---

## Windows 原生部署

如果 WSL2 不可用或需要更低延迟（无 GPU-PV 损耗），可在 Windows 上直接部署：

```cmd
:: 1. 安装 Python 3.13 + Ollama for Windows
::     https://www.python.org/downloads/
::     https://ollama.com/download/windows

:: 2. 设置 Ollama 环境变量
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_KEEP_ALIVE=24h

:: 3. 拉取模型
ollama pull qwen3:8b
ollama create qwen3:8b-nothink-v2 -f scripts\modelfile.qwen3-nothink-v2

:: 4. 克隆项目到本地（如 D:\project\nju-rule-rag）
git clone https://github.com/Mr-tree013/nju-rule-rag.git D:\project\nju-rule-rag
cd D:\project\nju-rule-rag

:: 5. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

:: 6. 配置 .env（同 Linux，LLM_BASE_URL=http://localhost:11434/v1）

:: 7. 构建索引
set PYTHONPATH=.
python scripts\build_chunks.py
python scripts\build_index.py

:: 8. 启动
start_windows.bat
```

Windows vs WSL2 延迟对比：实测两者相当（4.2-4.3s，差异在测量噪声内）。GPU-PV 对大模型推理无显著损耗。

---

## 在线端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 基本健康检查 |
| `/admin/health_deep` | GET | 深度健康检查（Ollama、GPU、模型、索引、缓存） |
| `/ask` | POST | 问答 `{"question": "..."}` |
| `/ask/stream` | POST | SSE 流式问答（三阶段：检索→生成→事实核验） |
| `/feedback` | POST | 用户反馈 `{"question","answer","rating":"up"/"down"}` |
| `/cache/stats` | GET | 缓存命中统计（内存 + 持久化） |
| `/admin/ingest_url` | POST | 通过 URL 录入新文档 |
| `/admin/staging` | GET | 列出暂存区文档 |
| `/qq` | POST | QQ Bot webhook |

### /ask 响应示例

```json
{
  "question": "补考没过怎么办？",
  "answer": "只能重修，补考就一次机会...",
  "risk_level": "medium",
  "need_human_confirm": false,
  "sources": [{"chunk_id", "source_id", "title", "url", "priority"}],
  "debug": {
    "retrieval_count": 40, "latency": 1.76,
    "llm_used": "qwen3:8b-nothink-v2",
    "cached": false,
    "confidence_tier": "1",
    "timing": {"classify_ms": 0, "retrieve_ms": 120, "rerank_ms": 350, "generate_ms": 1200}
  }
}
```

### /ask/stream SSE 事件

```
data: {"phase":"generating","chunks":20}
data: {"token":"补考"}
data: {"token":"没过"}
...
data: {"phase":"corrected","answer":"...","tier":"2"}   // fact_check 修改了答案时
data: {"done":true,"result":{...}}
```

---

## 架构

```
POST /ask {"question": "..."}
        │
        ▼
[_handle_meta_question]  "你是谁"/打招呼/晚安 → 直接回复（v0.7.0 扩展 casual chat）
        │
        ▼
TwoLayerRiskClassifier   L1关键词(高召回) → L2 BGE-M3 语义消歧
        │  🔒 GPU RLock
        ▼
[_classify_topic]        主题匹配 → soft boost 1.2×
        │
        ▼
HybridRetriever          BM25(0.25) + BGE-M3(0.45) + Priority(0.30)
        │  🔒 GPU RLock
        │  per-source max 3 chunks（防大杂烩文档垄断）
        ▼
CrossEncoderReranker     BGE-Reranker-v2-m3 (20候选 → 12精排)
        │  🔒 GPU Lock
        ▼
[QueryCache]             持久化 chunk 签名缓存（命中则跳过 LLM）
        │
_filter → _dedup         score阈值  → max 3/source, 12 total
        │
        ▼
[_decide_confidence]     3-tier: T1自信 → T2轻度hedge → T3跳过LLM
        │
        ▼
LLM (Qwen3-8B)           /api/generate 原生端点，/no_think 指令
        │  timeout→DeepSeek 回退，空响应→temperature 0.3 重试
        ▼
[fact_check]             NER实体校验 + COUNT_RE虚构数量检测 → 删除无出处句 / hedge / 降级T3
        │
        ▼
_format_response         600字截断 + 高风险模板
        │
        ▼
{ question, answer, risk_level, confidence_tier, sources[], debug }
```

---

## 模型清单

| 模型 | 大小 | 用途 | 线程安全 |
|------|------|------|---------|
| Qwen3-8B (no-think v2) | 5.2 GB | LLM 生成（Ollama `/api/generate`） | N/A（独立进程） |
| Qwen3-8B LoRA v3 | 5.0 GB | 微调 LLM（NJU QA 对，Q4_K_M GGUF） | N/A（独立进程） |
| BGE-M3 | 2.2 GB | 文本向量化（1024 维） | 否 — GPU RLock |
| BGE-Reranker-v2-m3 | 1.0 GB | 检索精排 | 否 — GPU Lock |
| BGE-Reranker-NJU | 1.0 GB | NJU 领域微调重排器 | 否 — GPU Lock |
| DeepSeek-Chat | API | 回退 LLM + 高风险路由 + 评测 judge | N/A |

总显存占用：~8-10 GB（Ollama + BGE 模型），16GB 余量充足。

---

## 评测体系

### 评测指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Faithfulness | **3.30/5** | 双 judge 平均（DeepSeek + Qwen3） |
| Relevance | **3.80/5** | 答案是否切题 |
| Recall@5 (rerank) | 88.1% | 正确文档在 top-5 的比例 |
| MRR | 0.612 | 平均倒数排名 |
| 关键词命中 | 77.8% | |
| 平均延迟 | **1.76s** | 从 5.03s 优化而来（-65%） |
| 端到端成功率 | 144/144 (100%) | |

### 评测脚本

```bash
python scripts/eval_rag.py                    # 端到端评测（需服务器）
PYTHONPATH=. python scripts/eval_retrieval.py # 检索指标（--rerank --rewrite）
PYTHONPATH=. python scripts/eval_generation.py --deepseek  # 忠实度/相关性评分
PYTHONPATH=. python scripts/eval_generation.py            # 本地 Qwen3 judge
PYTHONPATH=. python scripts/eval_compare_judges.py        # 双 judge 相关性分析
PYTHONPATH=. python scripts/error_taxonomy.py             # 失败模式分类
python scripts/annotate_gold_sources.py                   # 标注金标来源
```

### 人工校准

```bash
# 1. 生成人工抽检样本（25 题，覆盖 F=1~5）
#    样本已预生成在 data/eval/human_review_20q.json

# 2. 人工填写 human_F、human_R、human_notes 字段

# 3. 计算 Spearman 相关性
PYTHONPATH=. python scripts/eval_compare_judges.py
```

---

## 数据流水线

```bash
# 添加新文档到 data/processed/
# 更新 data/sources.csv 中的来源记录

# 重建索引
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py

# 反馈数据导出（积累训练数据）
python scripts/export_training_pairs.py              # 导出 👎 为训练对骨架
python scripts/review_feedback.py --stats            # 反馈统计
```

---

## 配置参考

```bash
# ── LLM ──
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b-nothink-v2

# LLM 回退
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_API_KEY=sk-xxx
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# ── 检索 ──
BM25_TOP_K=10; VECTOR_TOP_K=10; HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2; HIGH_RISK_MIN_SCORE=0.25

# ── 功能开关 ──
ENABLE_RERANK=true
ENABLE_FACT_CHECK=true
ENABLE_QUERY_REWRITE=false          # 口语改写（默认关闭）
ENABLE_CITATION_VERIFY=false        # bigram 校验（默认关闭）
ENABLE_HIGH_RISK_DEEPSEEK=true      # 高风险路由到 DeepSeek

# ── 重排 ──
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_CANDIDATE_K=20
RERANK_TOP_K=12
RERANKER_DEVICE=auto               # auto | cuda | cpu

# ── 置信度分级 ──
CONFIDENCE_TIER1_TOP1=0.65
CONFIDENCE_TIER1_TOP3=0.55
CONFIDENCE_TIER3_TOP1=0.25

# ── Prompt 预算 ──
PROMPT_TOKEN_BUDGET=6144
MAX_CHUNK_TOKENS=400
MAX_CHUNKS_IN_PROMPT=8

# ── Embedding ──
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3

# ── 显存管理 ──
EMPTY_CACHE_EVERY_N_REQUESTS=20
EMPTY_CACHE_FREE_VRAM_MB=1500

# ── QQ Bot ──
QQ_BOT_SELF_ID=你的QQ号
QQ_BOT_API_BASE_URL=http://127.0.0.1:8000
```

---

## 服务模块

| 文件 | 职责 |
|------|------|
| `app/main.py` | FastAPI 入口，所有端点 + CORS + 请求日志 |
| `app/pipeline.py` | RAGPipeline — 完整问答管线，每步骤可覆写 |
| `app/config.py` | Frozen Settings dataclass，.env 驱动 |
| `app/retriever.py` | HybridRetriever — BM25 + BGE-M3 向量 + 优先级加权 |
| `app/reranker.py` | CrossEncoderReranker — BGE-Reranker-v2-m3 精排 |
| `app/llm_client.py` | LLM 客户端 — `/api/generate` 原生端点 + 回退逻辑 |
| `app/fact_check.py` | NER 事实核验 — 数字/日期/URL/金额 比对 + COUNT_RE 虚构数量检测 + 三级惩罚 |
| `app/policy.py` | TwoLayerRiskClassifier + classify_topic + 响应模板 |
| `app/cache.py` | QACache (LRU) + PersistentQueryCache (chunk 签名) |
| `app/query_rewriter.py` | 口语查询改写（should_rewrite 守卫） |
| `app/health.py` | 深度健康检查 |
| `app/qq_bot.py` | QQ Bot 适配器（NapCat OneBot v11） |
| `app/deps.py` | 依赖注入 — 组件装配 |

---

## QQ Bot 接入

NapCat WebUI → 网络配置 → 新建 HTTP 客户端：

| 字段 | 值 |
|------|-----|
| 名称 | `nju-rule-rag` |
| URL | `http://127.0.0.1:8000/qq` |
| 消息格式 | `string` |

群内使用：`/问 补考没过怎么办` 或 `/ask 补考没过怎么办`

---

## 故障排查

### 启动报 "Network is unreachable"

WSL 终端残留 Windows 代理变量。`start_server.sh` 已自动处理，手动启动则需：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export HF_HUB_OFFLINE=1
```

### CUDA 不可用

```bash
python -c "import torch; print(torch.version.cuda)"  # 应显示 12.4
nvidia-smi | grep "CUDA Version"                       # 应一致
# 不一致则重装：pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 显存不足

```bash
RERANKER_DEVICE=cpu          # reranker 切 CPU（省 ~1GB）
ENABLE_RERANK=false          # 彻底关 reranker
ENABLE_VECTOR=false          # 纯 BM25（省 ~2.2GB）
```

### 构建命令报 "No module named 'app'"

需要 `PYTHONPATH=.` 前缀。

### 延迟突然飙升

显存碎片累积。重启服务即可恢复。

---

> 本系统仅提供一般性校规查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
