# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 61 份校规、办事指南和校园生活文档，支持自然语言提问、来源引用、风险分级与拒答机制。

**当前状态**：v0.2.0，端到端可运行，107 个测试通过，支持 Docker 部署。

---

## 快速开始

```bash
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                   # 编辑填入 LLM_API_KEY
uvicorn app.main:app --reload
```

**第一次运行会下载中文 embedding 模型（~400MB），请保持网络畅通。**

测试：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"缓考怎么申请？"}'
```

Docker 部署：

```bash
cp .env.example .env   # 编辑填入 API Key
docker compose up -d
```

---

## 项目结构

```
nju-rule-rag/
├── app/                          # 在线问答服务
│   ├── main.py                   # FastAPI 入口 (CORS, /health, /ask)
│   ├── config.py                 # Settings 数据类 (env → 冻结配置)
│   ├── errors.py                 # 异常层次 (RAGError 基类)
│   ├── deps.py                   # 依赖注入容器
│   ├── policy.py                 # 风险分类器 + 回答模板
│   ├── retriever.py              # BM25 + Chroma 混合检索 (Retriever 协议)
│   ├── llm_client.py             # LLM API 封装 (重试/脱敏)
│   ├── pipeline.py               # RAGPipeline (可组合步骤方法)
│   └── qq_bot.py                 # QQ Bot 适配层
│
├── scripts/                      # 离线数据处理
│   ├── build_chunks.py           # MD → 2075 chunks（条款切分）
│   ├── build_index.py            # chunks → BM25 + Chroma 索引
│   ├── validate_sources.py       # 校验 sources.csv
│   ├── validate_chunks.py        # 校验 chunks.jsonl
│   ├── crawl_sources.py          # 从 URL 抓取原始文件
│   ├── parse_documents.py        # 批量解析（sources.csv 驱动）
│   ├── parse_to_markdown.py      # 通用格式转换器
│   └── eval_rag.py               # 批量评测 /ask
│
├── tests/                        # 107 个测试
│   ├── test_answer_policy.py     # 37 个 — 风险分类 (旧 API + 新类)
│   ├── test_config.py            # 13 个 — Settings 数据类
│   ├── test_errors.py            # 5 个  — 异常层次
│   ├── test_retriever.py         # 22 个 — BM25/Vector/Hybrid
│   ├── test_llm_client.py        # 11 个 — LLMClient 初始化/脱敏
│   ├── test_pipeline.py          # 16 个 — RAGPipeline 步骤
│   └── test_main.py              # 5 个  — FastAPI 端点
│
├── data/
│   ├── sources.csv               # 61 条资料来源清单
│   ├── processed/                # 61 个 .md 文档 (本科生院 45 + 南哪助手 15 + 南京大学 1)
│   ├── chunks/                   # chunks.jsonl（2075条）+ stats
│   ├── index/                    # BM25 + Chroma + manifest
│   ├── eval/                     # 70 题评测集 + 结果
│   └── raw/                      # 原始抓取文件
│
├── docs/
│   ├── requirement.md            # MVP 需求文档
│   ├── risk_policy.md            # 风险分级策略
│   ├── source_priority.md        # 资料来源优先级
│   ├── dev_contract.md           # 开发契约（字段/接口/规则）
│   ├── evaluation_report.md      # 评测报告
│   └── demo_script.md            # Demo 10 问脚本
│
├── Dockerfile                    # Docker 镜像构建
├── docker-compose.yml            # 一键部署
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
└── README.md
```

---

## 模块状态

### app/ — 全部完成

| 文件 | 功能 |
|------|------|
| `main.py` | `/health`（含检索器状态、配置警告）、`/ask`，空问题 400，CORS 中间件 |
| `config.py` | `Settings` 冻结数据类，`RetrievalWeights`（含降级方案），`create_settings()` 工厂 |
| `errors.py` | `RAGError` 基类 → `ConfigError` / `LLMError` / `EmptyQuestionError` / `RetrievalError` |
| `deps.py` | `create_pipeline()` / `create_retriever()` / `create_llm_client()` 依赖装配 |
| `policy.py` | `RiskClassifier`（可子类扩展关键词），`ResponseTemplates`，`RiskLevel` 枚举 |
| `retriever.py` | `Retriever` 协议 + `BM25Retriever` / `VectorRetriever` / `HybridRetriever`。可注入分词器和权重 |
| `pipeline.py` | `RAGPipeline` 类，7 个可重写步骤方法，来源审计（年限/优先级），无全局单例 |
| `llm_client.py` | `LLMClient` + `EmbeddingClient`，3 次重试，Key 脱敏 |
| `qq_bot.py` | `/问` 触发，【结论】【依据】【提醒】三栏，从 Settings 读取配置 |

### scripts/ — 全部完成

| 文件 | 功能 |
|------|------|
| `build_chunks.py` | 第X条/一、/（一）/1.2.3. 切分，800cn 长分，30cn 短合，噪声过滤 |
| `build_index.py` | jieba BM25 + text2vec-base-chinese Chroma，ENABLE_VECTOR 可控 |
| `validate_sources.py` | 必要字段/唯一性/取值范围，exit 1 on error |
| `validate_chunks.py` | 字段/内容/唯一性，exit 1 on error |
| `crawl_sources.py` | 礼貌延迟 1s，跳过 need_login，失败不中断 |
| `parse_documents.py` | 批量解析：PDF(PyMuPDF) + DOC(LibreOffice) |
| `parse_to_markdown.py` | 通用转换器：HTML/PDF/DOC/DOCX/TXT → MD |
| `eval_rag.py` | 70 题批量评测，输出 results.csv + summary.json |

### data/ — 全部就绪

| 内容 | 数量 |
|------|------|
| 资料来源 | 61 条（本科生院 45 + 南哪助手 15 + 南京大学 1）|
| Markdown 文档 | 61 个 |
| 可检索片段 | 2075 chunks |
| 评测问题 | 70 题 / 14 主题 |

---

## 架构

```
POST /ask {"question": "..."}
        │
        ▼
RiskClassifier.classify()  →  ClassificationResult(level, is_process)
        │
        ▼
HybridRetriever.search()
  ├── BM25Retriever (jieba, bm25.pkl)
  └── VectorRetriever (ChromaDB, text2vec-base-chinese)
  Final = BM25_norm × 0.45 + vector_norm × 0.35 + priority_bonus × 0.20
        │
        ▼
_filter_chunks() → below MIN_RELIABLE_SCORE → refusal
        │         → high risk & below HIGH_RISK_MIN_SCORE → refusal
        │
        ▼
_build_prompt() → LLMClient.chat() → length cap → high-risk notice
        │                                            ↓
        ▼                              _audit_sources() (age / priority warnings)
_format_response()
        │
        ▼
{ question, answer, risk_level, need_human_confirm, sources[], debug }
```

### 设计原则

- **依赖注入**：`RAGPipeline` 通过构造函数接收 `Retriever`、`LLMClient`、`RiskClassifier`，无全局状态
- **协议接口**：`Retriever` 是 `Protocol`，任何实现 `.search()` 的对象都可接入
- **可扩展**：`RiskClassifier` 子类覆盖 `ClassVar` 元组即可添加关键词；`RAGPipeline` 子类覆盖步骤方法即可定制流程
- **优雅降级**：向量索引不可用时自动切换为纯 BM25；LLM 调用失败返回友好错误

---

## 完整数据流水线

```
┌─ 数据获取 ───────────────────────────────────────┐
│                                                    │
│  官网 / 通知频道              手动整理             │
│       │                         │                  │
│       ▼                         ▼                  │
│  crawl_sources.py         直接放入 processed/      │
│  (下载 HTML/PDF)           (*.md)                  │
│       │                                            │
│       ▼                                            │
│  parse_to_markdown.py  ←── 通用格式转换器           │
│  HTML/PDF/DOC/DOCX → MD                            │
│       │                                            │
│       ▼                                            │
│  data/processed/*.md  ←── 61 个清洗后文档          │
│                                                    │
└───────────────────┬────────────────────────────────┘
                    │
┌─ 索引构建 ──────────────────────────────────────┐
│                                                    │
│  build_chunks.py                                   │
│  按条款切分 → 长段拆分 → 短段合并 → 噪声过滤       │
│       │                                            │
│       ▼                                            │
│  chunks.jsonl (2075 chunks) + chunk_stats.json       │
│       │                                            │
│       ▼                                            │
│  build_index.py                                    │
│  BM25(jieba) + Chroma(text2vec-base-chinese)       │
│       │                                            │
│       ▼                                            │
│  index/bm25.pkl + chunk_lookup.json + manifest.json│
│  index/chroma/ (向量存储)                          │
│                                                    │
└───────────────────┬────────────────────────────────┘
                    │
┌─ 在线问答 ──────────────────────────────────────┐
│                                                    │
│  POST /ask {"question": "..."}                     │
│       │                                            │
│       ▼                                            │
│  RiskClassifier.classify() → low / medium / high   │
│       │                                            │
│       ▼                                            │
│  HybridRetriever: BM25(0.45)+Vector(0.35)+Pri(0.20)│
│       │                                            │
│       ▼                                            │
│  分数过滤 → 低于阈值 → 拒答                         │
│       │                                            │
│       ▼                                            │
│  LLM 生成 → 长度截断 → 高风险追加提醒               │
│       │         → _audit_sources() (年限/优先级)    │
│       ▼                                            │
│  { question, answer, risk_level,                    │
│    need_human_confirm, sources, debug }             │
│                                                    │
└────────────────────────────────────────────────────┘
```

**文档更新后重建**：

```bash
python scripts/build_chunks.py
python scripts/build_index.py
python scripts/validate_sources.py
python scripts/validate_chunks.py
```

---

## 引入新文档

```bash
# 方式一：有 URL → 自动抓取
# 1. 在 data/sources.csv 加一行（填好 url/source_type/priority）
python scripts/crawl_sources.py
python scripts/parse_to_markdown.py data/raw/nju-jw-xxx.html \
  --title "通知标题" -o data/processed/新文档.md

# 方式二：有原始文件 → 直接转换
python scripts/parse_to_markdown.py ~/下载/通知.pdf \
  --title "通知标题" -o data/processed/新文档.md

# 方式三：手动整理 Markdown → 直接放 data/processed/

# 都完成后重建索引
python scripts/build_chunks.py && python scripts/build_index.py
```

---

## 配置

```bash
# .env（必填）
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# 检索调参（可选）
BM25_TOP_K=10
VECTOR_TOP_K=10
HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2
HIGH_RISK_MIN_SCORE=0.25

# 构建选项
ENABLE_VECTOR=true
LOCAL_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

支持所有 OpenAI 兼容接口（DeepSeek / 通义千问 / 智谱 / OpenAI）。

---

## 评测

```bash
# 确保服务运行
uvicorn app.main:app --reload &

# 跑 70 题评测
python scripts/eval_rag.py

# 查看汇总
python -m json.tool data/eval/summary.json
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 接口 | FastAPI + Uvicorn |
| 关键词检索 | BM25 (rank-bm25) + jieba |
| 语义检索 | ChromaDB + text2vec-base-chinese |
| 文本生成 | OpenAI 兼容 API |
| 风控 | policy.py（关键词+规则+来源审计）|
| 格式转换 | PyMuPDF + BeautifulSoup + LibreOffice |
| 部署 | Docker + docker-compose |
| 测试 | pytest (107 tests) |

---

## 已知限制

1. **向量索引依赖网络**：首次运行需下载 text2vec-base-chinese（~400MB），离线环境需预下载或设置 `ENABLE_VECTOR=false`
2. **关键词误报**：含"学位"的非高风险问题偶尔被判为 medium（已从 high 降级），"学位证"相关信息性查询仍会触发 `need_human_confirm`
3. **学生手册 25 段 >800 中文字符**：连续表格，无段落边界可切
4. **DeepSeek API 偶发超时**：已加 3 次重试，极端情况下仍可能失败
5. **3 份教务系统手册以截图为主**：文字量少，检索效果差
6. **评测数据需更新**：`data/eval/summary.json` 是旧版无 LLM 运行结果，需配置 API Key 后重新跑

---

## 下一步

| 优先级 | 任务 | 状态 |
|--------|------|------|
| **P0** | QQ Bot 接入实际框架（NapCat/ Lagrange 等）测试 | 待完成 |
| **P0** | 通知频道自动抓取 + 增量索引（定时 crawl + diff） | 待完成 |
| **P0** | 向量索引修复：在有网络环境下重跑 `build_index.py`，恢复混合检索双路 | 待完成 |
| **P1** | 优化融合权重 + 通过 `.env` 暴露权重参数 | 待完成 |
| **P1** | 重跑 70 题评测（需配好 LLM API Key），更新 `data/eval/` | 待完成 |
| **P1** | 嵌入模型升级评估：`text2vec-base-chinese` → `bge-m3` | 待完成 |
| **P2** | 部署供小范围试用（云服务器 / 校内服务器） | 待完成 |
| **P2** | CI/CD：GitHub Actions 自动测试 + 构建 Docker 镜像 | 待完成 |
| **P2** | 前端界面：简易 Web 问答页面或小程序 | 待完成 |
| **持续** | 追踪本科生院通知更新，增量添加新文档 | 持续 |
| **持续** | 根据用户反馈扩充高风险 case，调优关键词 | 持续 |

---

> 本系统仅提供一般性校规和生活信息查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
>
> A（在线服务）× B（离线数据）× 契约（chunks.jsonl / manifest.json / POST /ask）
