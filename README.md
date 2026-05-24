# NJU Rule RAG

南京大学本科新生校规与教务流程 RAG（检索增强生成）问答系统。

项目目标：完成一个可演示、可小范围试用的初步成果，支持本地问答、FastAPI `/ask`、QQ Bot、来源引用、风险分级与拒答机制。

团队规模：2 人。分工原则：一人负责 `app/` 在线服务侧，一人负责 `scripts/` 与 `data/` 离线数据侧。

---

## 第 1 周已完成

| 阶段 | 产出 | 状态 |
|------|------|------|
| MVP 范围定义 | `docs/requirement.md` `docs/risk_policy.md` `docs/source_priority.md` | ✅ |
| 项目环境搭建 | FastAPI + venv + 依赖，`/health` `​/ask` 接口可用 | ✅ |
| 资料编目 | `data/sources.csv` — 26 个来源，Priority 1-5 分级 | ✅ |
| 文档解析 | 24 个 `.md` 文件（23 份现行 2025 版校规 + 2021 版学生手册） | ✅ |
| 文本切分 | **396 chunks**，按条款/章节切分，含完整元数据 | ✅ |
| 检索索引 | BM25 关键词索引 + Chroma 向量索引（ONNX 本地 embedding） | ✅ |

### MVP 覆盖主题（15 个）

选课 · 缓考 · 补考 · 重修 · 成绩管理 · 学籍异动 · 辅修 · 交换培养 · 学业预警 · 考试纪律与处分 · 学生申诉 · 推免研究生 · 学分学费 · 奖学金与助学 · 常见办事流程

---

## 模块状态总览

### A 同学：App 在线问答侧

| 文件 | 功能 | 状态 | 优先级 |
|------|------|------|--------|
| `app/main.py` | FastAPI 入口，`/health` `​/ask` 接口 | ✅ | P0 |
| `app/config.py` | `.env` 配置读取 | ✅ | P0 |
| `app/answer_policy.py` | 风险分类、拒答、高风险提示 | ⏳ | P0 |
| `app/retriever.py` | 在线检索与混合排序（BM25 + Chroma） | ⏳ | P0 |
| `app/llm_client.py` | LLM 与 Embedding API 封装 | ⏳ | P0 |
| `app/rag_pipeline.py` | 串联检索→LLM→风险分级的完整 RAG 流程 | ⏳ | P0 |
| `app/qq_bot.py` | QQ Bot 适配层（只调 `/ask`，不写 RAG 逻辑） | ⏳ | P1 |

### B 同学：Scripts + Data 离线数据侧

| 文件 | 功能 | 状态 | 优先级 |
|------|------|------|--------|
| `data/sources.csv` | 26 个资料源清单 | ✅ | P0 |
| `data/processed/` | 24 个 `.md` 清洗后文档 | ✅ | P0 |
| `scripts/build_chunks.py` | `.md` 按条款切分为 396 chunks | ✅ | P0 |
| `scripts/build_index.py` | 构建 BM25 + Chroma 双索引 | ✅ | P0 |
| `data/chunks/chunks.jsonl` | 396 条 chunk，含完整元数据 | ✅ | P0 |
| `data/index/` | BM25 (pickle) + Chroma (sqlite3) 索引 | ✅ | P0 |
| `scripts/validate_sources.py` | 校验 sources.csv 完整性 | ⏳ | P0 |
| `scripts/validate_chunks.py` | 校验 chunks.jsonl 字段完整性 | ⏳ | P0 |
| `scripts/crawl_sources.py` | 抓取网页/PDF 到 `data/raw/` | ⏳ | P1 |
| `scripts/parse_documents.py` | HTML/PDF → Markdown | ⏳ | P1 |
| `scripts/eval_rag.py` | 批量评测 `/ask` 接口 | ⏳ | P1 |
| `data/eval/questions.csv` | 评测问题集（≥ 50 条） | ⏳ | P1 |

---

## 三大协作契约

两人并行开发前，必须冻结以下三个契约。

### 契约一：`data/chunks/chunks.jsonl`

每个 chunk 至少包含：

```json
{
  "chunk_id": "nju-jw-001-0001",
  "source_id": "nju-jw-001",
  "title": "南京大学本科生缓考管理相关文件",
  "url": "https://example.com",
  "department": "本科生院",
  "scope": "本科生",
  "priority": 1,
  "article": "第三条",
  "content": "第三条 ……",
  "fetched_at": "2026-05-23 10:00:00"
}
```

| 字段 | 说明 | 必需 |
|------|------|------|
| `chunk_id` | 唯一 ID，稳定可复现 | 是 |
| `source_id` | 来源 ID，对应 `sources.csv` | 是 |
| `title` | 来源标题 | 是 |
| `url` | 来源 URL | 是 |
| `department` | 发布部门 | 是 |
| `scope` | 适用范围 | 是 |
| `priority` | 优先级 1-5 | 是 |
| `article` | 条款号或小标题 | 建议保留 |
| `content` | chunk 正文 | 是 |
| `fetched_at` | 抓取时间 | 是 |

### 契约二：`data/index/manifest.json`

```json
{
  "built_at": "2026-05-24 20:00:00",
  "chunks_file": "data/chunks/chunks.jsonl",
  "chunk_count": 382,
  "bm25_index": "data/index/bm25.pkl",
  "chunk_lookup": "data/index/chunk_lookup.json",
  "vector_index": "data/index/chroma",
  "embedding_model": "bge-m3",
  "status": "ok"
}
```

- `app/retriever.py` 启动时读取 `manifest.json`
- 向量索引不可用时，优雅降级到 BM25
- `build_index.py` 不破坏已有可用索引

### 契约三：`POST /ask` 返回格式

请求：

```json
{ "question": "缓考怎么申请？" }
```

响应：

```json
{
  "question": "缓考怎么申请？",
  "answer": "……",
  "risk_level": "medium",
  "need_human_confirm": true,
  "sources": [
    {
      "chunk_id": "nju-jw-001-0001",
      "source_id": "nju-jw-001",
      "title": "南京大学本科生缓考管理相关文件",
      "url": "https://example.com",
      "priority": 1
    }
  ],
  "debug": {
    "retrieval_count": 5,
    "latency": 2.31
  }
}
```

- B 的 `eval_rag.py` 只调 `/ask`，A 的 `qq_bot.py` 也只调 `/ask`
- `qq_bot.py` 不直接调 `rag_pipeline.py`
- `sources` 只能来自检索到的 chunks
- 找不到依据时 `sources` 可以为空，但 `answer` 必须明确说明依据不足

---

## 六天推进流程

| 天数 | A 同学（App 侧） | B 同学（Data 侧） | 联调验收 |
|------|------------------|-------------------|----------|
| **Day 1** | `config.py` + `main.py` + `answer_policy.py`；`/ask` 返回 mock 结构 | `validate_sources.py` + `validate_chunks.py`；创建 `docs/dev_contract.md` | `/health` 正常，`/ask` 返回固定字段 |
| **Day 2** | `retriever.py`（BM25 检索） | `build_index.py` 生成 BM25 索引 + `manifest.json` | Top 5 至少 1 个相关 chunk，带来源标题 |
| **Day 3** | `llm_client.py` + `rag_pipeline.py`（完整 RAG） | 优化 `build_chunks.py` + chunks 质量检查 | 能回答、有来源、高风险不直接下结论 |
| **Day 4** | 保证 `/ask` 稳定可批量调用 | `questions.csv`（≥50 条）+ `eval_rag.py` | 跑完整评测，输出 `summary.json` |
| **Day 5** | 修复 Top 10 bad cases（retriever/pipeline/policy/prompt） | 修复数据侧 bad cases（chunk 噪声/缺资料） | 修复前后对比，重新跑评测 |
| **Day 6** | `qq_bot.py` + Demo 准备 | `evaluation_report.md` + `demo_script.md` | Demo 10 问 ≥ 8 问稳定 |

---

## 技术栈一览

```
                    用户提问
                       │
                       ▼
              ┌─ main.py /ask ─┐
              │    (FastAPI)    │
              └───────┬────────┘
                      │
         ┌─────────rag_pipeline ────────┐
         │          (⏳待开发)           │
         ▼             ▼               ▼
   retriever.py   llm_client.py  answer_policy.py
   (⏳待开发)     (⏳待开发)      (⏳待开发)
         │
    ┌────┴────┐
    ▼         ▼
  BM25     Chroma
 (pickle)  (sqlite3)
    │         │
    └────┬────┘
         ▼
   混合排序 Top-K
         │
         ▼
    相关校规片段  ──→  LLM 生成答案  ──→  风险分级  ──→  返回 JSON
```

| 层 | 技术 | 用途 |
|----|------|------|
| 接口 | FastAPI + Uvicorn | REST API 服务 |
| 检索-关键词 | BM25 (rank-bm25) + jieba 分词 | 字面匹配校规条款 |
| 检索-语义 | ChromaDB + ONNX (all-MiniLM-L6-v2) | 语义相似度搜索 |
| 生成 | OpenAI / DeepSeek / 通义千问 API | 基于检索结果生成回答 |
| 风控 | `answer_policy.py` | 风险分级 + 敏感问题拒答 |
| 文档解析 | Python 脚本 + 人工整理 | HTML/PDF → Markdown |

### 关键模块接口速查

**`answer_policy.py`**
```python
classify_question(question: str) -> str          # low / medium / high
is_process_question(question: str) -> bool        # 是否流程类问题
need_human_confirm(question: str, risk_level: str) -> bool
no_evidence_response(question: str) -> dict       # 标准拒答
build_high_risk_notice(question: str) -> str      # 高风险提示
```

**`retriever.py`**
```python
class BM25Retriever:
    def search(self, question: str, top_k: int = 10) -> list[dict]

class VectorRetriever:
    def search(self, question: str, top_k: int = 10) -> list[dict]

class HybridRetriever:
    def search(self, question: str, top_k: int = 5) -> list[dict]
    # final_score = bm25_norm * 0.5 + vector_norm * 0.4 + priority_score * 0.1
```

**`llm_client.py`**
```python
chat(messages: list[dict], temperature: float = 0.2) -> str
embed_texts(texts: list[str]) -> list[list[float]]
```

**`rag_pipeline.py`**
```python
answer_question(question: str) -> dict
build_context(chunks: list[dict]) -> str
build_prompt(question: str, chunks: list[dict], risk_level: str) -> list[dict]
```

---

## 目录结构

```
nju-rule-rag/
├── app/                        # A 同学：在线问答服务
│   ├── main.py                 # ✅ FastAPI 入口 (/health, /ask)
│   ├── config.py               # ✅ .env 配置读取
│   ├── answer_policy.py        # ⏳ 风险分类与拒答
│   ├── retriever.py            # ⏳ 混合检索
│   ├── llm_client.py           # ⏳ LLM 调用封装
│   ├── rag_pipeline.py         # ⏳ RAG 总流程
│   └── qq_bot.py               # ⏳ QQ Bot 适配
│
├── scripts/                    # B 同学：离线数据处理
│   ├── build_chunks.py         # ✅ .md → 396 chunks
│   ├── build_index.py          # ✅ chunks → BM25 + Chroma 索引
│   ├── validate_sources.py     # ⏳ 校验 sources.csv
│   ├── validate_chunks.py      # ⏳ 校验 chunks.jsonl
│   ├── crawl_sources.py        # ⏳ 抓取网页/PDF
│   ├── parse_documents.py      # ⏳ HTML/PDF 解析
│   └── eval_rag.py             # ⏳ 批量评测 /ask
│
├── data/                       # 数据（已纳入 Git）
│   ├── sources.csv             # ✅ 26 个资料来源清单
│   ├── processed/              # ✅ 24 个 .md 校规文件
│   ├── chunks/                 # ✅ chunks.jsonl（396 条）
│   ├── index/                  # ✅ BM25 + Chroma 索引
│   │   ├── bm25_index.pkl
│   │   ├── chunk_lookup.json  ⏳
│   │   ├── manifest.json      ⏳
│   │   └── chroma/
│   ├── raw/                    # 原始 HTML/PDF（当前跳过）
│   └── eval/                   # 评测数据（待构建）
│
├── docs/                       # 设计文档
│   ├── requirement.md          # ✅ MVP 需求文档
│   ├── risk_policy.md          # ✅ 风险分级策略
│   ├── source_priority.md      # ✅ 资料来源优先级
│   └── dev_contract.md         # ⏳ 开发契约
│
├── tests/                      # 测试（待编写）
├── .env.example                # 环境变量模板
├── requirements.txt            # Python 依赖（9 个包）
└── README.md
```

---

## Git 协作规范

### 分支建议

```text
main                     # 稳定可演示版本
dev                      # 日常联调分支
feature/app-rag          # A 同学：RAG / API
feature/app-qq-bot       # A 同学：QQ Bot
feature/data-pipeline    # B 同学：资料抓取、解析、chunk、索引
feature/data-eval        # B 同学：评测集与评测脚本
```

- 不要直接推 `main`
- 每天至少一次合并到 `dev`
- `main` 只保留稳定可演示版本

### Commit 格式

```text
feat(app): add answer policy
feat(data): add chunk validation script
fix(rag): prevent source hallucination
fix(data): remove noisy footer chunks
test(app): add policy tests
docs: add dev contract
```

### 每日合并前检查

```bash
python scripts/validate_sources.py
python scripts/validate_chunks.py
python scripts/build_index.py
uvicorn app.main:app --reload
python scripts/eval_rag.py
```

### PR 检查清单

**A 同学：**
- [ ] `/health` 正常
- [ ] `/ask` 返回固定字段
- [ ] 空问题有错误处理
- [ ] 高风险问题不直接下结论
- [ ] `sources` 只来自 chunks
- [ ] 没有提交 `.env` 或 API key

**B 同学：**
- [ ] `validate_sources.py` 通过
- [ ] `validate_chunks.py` 通过
- [ ] `chunks.jsonl` 字段完整
- [ ] `chunk_stats.json` 已更新
- [ ] `build_index.py` 能跑完
- [ ] `manifest.json` 已生成

---

## 最小成功版本

如果时间不够，优先砍掉向量检索和 QQ Bot，保留：

```
BM25 + 风险策略 + 来源引用 + FastAPI /ask + 评测
```

| 角色 | 最低交付 |
|------|----------|
| A | `main.py` + `config.py` + `answer_policy.py` + `retriever.py` + `rag_pipeline.py` + `llm_client.py` |
| B | `sources.csv` + `chunks.jsonl` + `validate_chunks.py` + `build_index.py` + `questions.csv`(≥50) + `eval_rag.py` |

---

## 最终验收顺序

1. B：`validate_sources.py` 通过
2. B：`validate_chunks.py` 通过
3. B：`build_index.py` 生成 `manifest.json`
4. A：`/health` 正常
5. A：`/ask` 对 5 个问题返回结构化 JSON
6. A+B：`eval_rag.py` 跑 50 条
7. A+B：修复 Top 10 bad cases
8. A：QQ Bot 接入 `/ask`
9. A+B：Demo 10 问
10. 写 README 和 `evaluation_report.md`

---

## 快速开始

```bash
# 1. 克隆（仓库已含预处理数据，无需额外构建）
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag

# 2. 虚拟环境 + 依赖
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. 配置文件（后续用到 LLM 时再填 key）
cp .env.example .env

# 4. 启动服务
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/health → {"status": "ok"}
```

### 文档更新后重建索引

```bash
source .venv/bin/activate
python scripts/build_chunks.py    # .md → chunks.jsonl
python scripts/build_index.py     # chunks → BM25 + Chroma
```

---

## 风险提示

本系统仅提供校规和流程的一般性查询，不替代学院教务员、辅导员或相关部门的正式答复。涉及退学、处分、作弊、学位、毕业资格等高风险问题，系统不会直接给出个人结论。

> A 负责"服务稳定、回答可信、风险可控、Bot 可用"；B 负责"资料可靠、chunk 可用、索引可复现、评测能闭环"。两人通过 `chunks.jsonl`、`manifest.json`、`/ask` 三个契约协作。
