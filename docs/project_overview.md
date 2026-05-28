# NJU Rule RAG 项目框架介绍

## 一、项目简介

NJU Rule RAG 是一个面向南京大学本科新生的**校规与校园生活智能问答系统**。基于 60+ 份公开校规、办事指南和校园生活文档，支持自然语言提问、来源引用、风险分级与拒答机制。提供 FastAPI 接口和 QQ 群机器人两种交互方式。

**目标用户**：南京大学本科新生（大一至大二）。

---

## 二、项目规模

| 维度 | 数值 |
|------|------|
| 资料来源 | 60+ 份文档（校规 38 + 通知 10 + 生活指南 15+） |
| 可检索片段 | 800+ chunks |
| 核心代码 | 约 3000 行 Python |
| 单元测试 | 107+ 个 |
| 评测问题 | 70 题 / 14 个主题 |
| Docker 支持 | ✅ 一条命令部署 |

---

## 三、整体架构

```
┌────────────────────────────────────────────────────┐
│                    用户层                           │
│     QQ 群 @机器人          curl / Web 前端          │
│         │                       │                  │
├─────────┼───────────────────────┼──────────────────┤
│  接入层 │  NapCatQQ              │  FastAPI         │
│         │  (OneBot v11 HTTP)     │  /ask /health /qq│
│         │         │              │                  │
├─────────┴─────────┼──────────────┴──────────────────┤
│                    │                                 │
│              ┌─────▼──────┐                         │
│              │  pipeline  │  核心编排                │
│              │  风险分级的  │                         │
│              │  RAG 全流程 │                         │
│              └─┬────────┬─┘                         │
│         ┌──────┘        └──────┐                    │
│    ┌────▼────┐           ┌─────▼─────┐              │
│    │ policy  │           │ retriever │              │
│    │ 风险分类 │           │ 混合检索   │              │
│    │ 拒答模板 │           │ BM25+向量  │              │
│    └─────────┘           │ +优先级加成 │              │
│                          └─────┬─────┘              │
│                                │                    │
│                   ┌────────────┼────────────┐       │
│              ┌────▼────┐  ┌────▼────┐  ┌───▼───┐  │
│              │ llm_clie │  │ChromaDB │  │BM25   │  │
│              │ (LLM API)│  │(向量库) │  │(关键词)│  │
│              └──────────┘  └─────────┘  └───────┘  │
│                                                      │
│    ┌──────────── 离线数据层 ────────────┐            │
│    │  文档 → 格式转换 → MD → 切分chunks → 建索引 │    │
│    │  sources.csv    parse/ build_*   build_index │    │
│    └────────────────────────────────────┘            │
│                                                      │
│    ┌──────────── 质量保障层 ────────────┐            │
│    │  validate_sources  validate_chunks             │
│    │  eval_rag  eval_retrieval  eval_generation     │
│    │  check_regression  (107+ 自动化测试)           │
│    └────────────────────────────────────┘            │
└────────────────────────────────────────────────────┘
```

**设计原则**：依赖注入、协议接口、可插拔扩展、优雅降级。

---

## 四、核心模块

### `app/` 在线服务层

| 文件 | 职责 | 行数 |
|------|------|------|
| `main.py` | FastAPI 入口，`/health` `/ask` `/qq` 三个端点，CORS 中间件 | ~140 |
| `pipeline.py` | RAG 全流程编排：分类→检索→过滤→LLM→格式化 | ~220 |
| `policy.py` | 风险分类器 + 回答模板，支持子类扩展关键词 | ~190 |
| `retriever.py` | BM25 + Chroma 混合检索，权重 0.45/0.35/0.20，优先级加成 | ~470 |
| `reranker.py` | 检索结果重排序，提升答案准确性 | ~60 |
| `llm_client.py` | LLM API 封装，自动重试 + Key 脱敏，支持 OpenAI 兼容接口 | ~240 |
| `qq_bot.py` | QQ Bot 适配层，`/问` 触发，【结论】【依据】【提醒】三段式回复 | ~110 |
| `config.py` | 冻结数据类 Settings，所有阈值/路径/模型名从 `.env` 读取 | ~235 |
| `deps.py` | 依赖注入容器，统一装配所有组件 | ~90 |
| `errors.py` | 统一异常层次：`RAGError` 基类 | ~30 |

### `scripts/` 离线数据层

| 文件 | 职责 |
|------|------|
| `crawl_sources.py` | 从 URL 抓取原始 HTML/PDF |
| `parse_to_markdown.py` | 通用格式转换器（HTML/PDF/DOC/DOCX/TXT → MD） |
| `parse_documents.py` | 批量解析（PyMuPDF + LibreOffice） |
| `build_chunks.py` | MD → chunks，按条款切分 + 长分短合 + 噪声过滤 |
| `build_index.py` | chunks → BM25 + Chroma 向量索引 |
| `validate_sources.py` | 校验 sources.csv 字段/唯一性/取值范围 |
| `validate_chunks.py` | 校验 chunks.jsonl 字段/内容/唯一性 |

### `scripts/` 评测层

| 文件 | 职责 |
|------|------|
| `eval_rag.py` | 端到端 `/ask` 评测（70 题） |
| `eval_retrieval.py` | 检索质量独立评测（有/无 rerank 对照） |
| `eval_generation.py` | 生成质量独立评测（多模型对比） |
| `annotate_gold_sources.py` | 标注黄金标准来源 |
| `check_regression.py` | 回归测试，防止性能回退 |

---

## 五、数据流水线

```
┌─ 数据获取 ─────────────────────────────────────────────┐
│                                                          │
│  公开网页/PDF              手动整理 Markdown             │
│       │                         │                        │
│       ▼                         ▼                        │
│  crawl_sources.py         直接放入 processed/            │
│  (下载原始文件)            (*.md)                        │
│       │                                                  │
│       ▼                                                  │
│  parse_to_markdown.py  ←── 通用格式转换器                 │
│  HTML/PDF/DOC/DOCX → MD                                  │
│       │                                                  │
│       ▼                                                  │
│  data/processed/*.md  ←── 60+ 个清洗后文档               │
│                                                          │
└───────────────────┬──────────────────────────────────────┘
                    │
┌─ 索引构建 ────────────────────────────────────────────┐
│                                                          │
│  build_chunks.py                                         │
│  第X条/一、/（一）/##/### 切分 → 800字长分 → 30字短合 → 噪声过滤 │
│       │                                                  │
│       ▼                                                  │
│  chunks.jsonl (800+ chunks)                              │
│       │                                                  │
│       ▼                                                  │
│  build_index.py                                          │
│  BM25 (jieba) + ChromaDB (text2vec-base-chinese)        │
│       │                                                  │
│       ▼                                                  │
│  index/bm25.pkl + chunk_lookup.json + manifest.json     │
│  index/chroma/ (向量存储)                                │
│                                                          │
└───────────────────┬──────────────────────────────────────┘
                    │
┌─ 在线问答 ────────────────────────────────────────────┐
│                                                          │
│  用户提问                                                │
│       │                                                  │
│       ▼                                                  │
│  policy.RiskClassifier  →  low / medium / high          │
│       │                                                  │
│       ▼                                                  │
│  HybridRetriever  →  BM25(0.45) + Vector(0.35) + Pri(0.20) │
│       │                                                  │
│       ▼                                                  │
│  reranker.Reranker  →  检索结果重排序                     │
│       │                                                  │
│       ▼                                                  │
│  分数过滤  →  低于阈值  →  拒答                            │
│       │                                                  │
│       ▼                                                  │
│  LLM 生成  →  长度截断  →  高风险追加提醒                 │
│       │                                                  │
│       ▼                                                  │
│  { question, answer, risk_level, sources, debug }       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 六、技术栈

| 层 | 技术 |
|----|------|
| Web 接口 | FastAPI + Uvicorn |
| 关键词检索 | BM25 (rank-bm25) + jieba 中文分词 |
| 语义检索 | ChromaDB + text2vec-base-chinese |
| 重排序 | app/reranker.py |
| 文本生成 | OpenAI 兼容接口（支持 DeepSeek / Qwen / 智谱 / Ollama 本地模型） |
| 风险控制 | policy.py（关键词分类 + 规则 + 来源审计） |
| 格式转换 | PyMuPDF + BeautifulSoup + LibreOffice + python-docx |
| QQ Bot | NapCatQQ + OneBot v11 HTTP 回调 |
| 评测 | pytest (107+ tests) + 独立评测脚本 |
| 部署 | Docker + docker-compose |

---

## 七、部署方式

### 方式一：本地部署（开发/测试）

```bash
pip install -r requirements.txt
cp .env.example .env          # 编辑填入 LLM_API_KEY
uvicorn app.main:app --reload
```

### 方式二：Docker 部署

```bash
cp .env.example .env          # 编辑填入 API Key
docker compose up -d
```

### 方式三：云服务器部署（生产/QQ Bot）

使用 AutoDL 等 GPU 云平台租用服务器，部署 Ollama 本地大模型 + FastAPI 后端 + 官方 QQ Bot SDK，一台服务器运行全部服务。每月成本约 300 元（RTX 3090 包月）。

### QQ Bot 接入方式

项目提供两种 QQ Bot 接入方案：

| 方案 | 原理 | 适用场景 |
|------|------|---------|
| **NapCatQQ + QQ 小号** | 注册 QQ 小号 → 扫码登录 NapCat → OneBot v11 HTTP 回调到 FastAPI `/qq` 端点 | 快速 demo，零成本，群消息全可见 |
| 官方 QQ Bot SDK | 使用 QQ 官方开放平台 AppID/Secret → WebSocket 直连腾讯服务器 | 生产部署，无封号风险，群消息仅 @机器人 时触发 |

**推荐 demo 方案**：NapCatQQ + 注册一个新的 QQ 小号，扫码登录后 NapCat 自动转发群消息到后端。无需审核、零成本、可立即在 QQ 群中演示。

```
QQ 群 @机器人 /问 缓考怎么申请
       │
       ▼
NapCatQQ（QQ小号后台运行）
       │  OneBot v11 HTTP POST
       ▼
app/main.py → /qq 端点
       │
       ▼
app/qq_bot.handle_message()  →  RAG 全流程
       │
       ▼
返回【结论】【依据】【提醒】
       │
       ▼  NapCat 发回 QQ 群
```

---

## 八、快速开始

```bash
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
cp .env.example .env              # 填入 LLM_API_KEY
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

运行评测：
```bash
# 确保服务已启动
python scripts/eval_rag.py
python scripts/eval_retrieval.py
python scripts/eval_generation.py
```

运行测试：
```bash
pytest tests/ -v
```

---

## 九、已知限制

1. 向量索引首次运行需下载中文模型（~400MB），离线环境需预下载
2. 含"学位"关键词的非高风险问题偶尔触发 `need_human_confirm`
3. 部分教务系统手册以截图为主，文字量少，检索效果差
4. LLM API 偶发超时（已加 3 次重试 + 指数退避）

---

> 本系统仅提供一般性校规和生活信息查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
