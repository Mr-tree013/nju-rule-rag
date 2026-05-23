# NJU Rule RAG

南京大学本科新生校规与教务流程 RAG（检索增强生成）问答系统。

## 当前进度

### 第 1 周已完成

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

### 待开发（第 2 周）

- `app/retriever.py` — 封装 BM25 + Chroma 检索，混合排序
- `app/llm_client.py` — LLM API 调用（DeepSeek / 通义千问 / OpenAI）
- `app/answer_policy.py` — 风险分级 + 拒答模板
- `app/rag_pipeline.py` — 串联检索→LLM→策略的完整问答流程

---

## 技术架构

```
用户提问 → 混合检索(BM25+向量) → LLM生成答案 → 风险分级 → 返回{答案, 来源, 风险等级}
```

- **后端框架**: FastAPI + Uvicorn
- **检索**: BM25 (rank-bm25 + jieba) + 向量检索 (Chroma, 本地 ONNX embedding)
- **LLM**: 可替换接口（支持 OpenAI / DeepSeek / 通义千问 / 智谱）
- **文档处理**: PyMuPDF + pdfplumber + BeautifulSoup

---

## 快速开始（团队成员）

```bash
# 1. 克隆仓库
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. 配置文件（开发阶段可暂不填 LLM key）
cp .env.example .env

# 4. 启动服务
uvicorn app.main:app --reload
# 访问 http://127.0.0.1:8000/health → {"status": "ok"}
```

---

## 数据处理流程

由于文档已经以 `.md` 格式提供，**跳过抓取和 PDF 解析**，直接从切分开始：

```bash
source .venv/bin/activate

# Step 1: 文档切分
# 读取 data/processed/*.md → 按条款/章节切分 → data/chunks/chunks.jsonl
python scripts/build_chunks.py

# Step 2: 构建索引
# 读取 chunks.jsonl → 构建 BM25 + Chroma 向量索引 → data/index/
python scripts/build_index.py
```

预期输出：24 个文件 → 396 chunks → 双索引就绪。

### 如需新增文档

1. 将 `.md` 文件放入 `data/processed/`
2. 在 `data/sources.csv` 中添加一行（指定 filename 字段匹配文件名）
3. 重新运行上述两个脚本

### 如需从零开始（有新的 HTML/PDF 源文件）

1. 编辑 `data/sources.csv`
2. 运行 `python scripts/crawl_sources.py`（待开发）
3. 运行 `python scripts/parse_documents.py`（待开发）
4. 运行 `python scripts/build_chunks.py`
5. 运行 `python scripts/build_index.py`

---

## 目录结构

```
nju-rule-rag/
├── app/
│   ├── main.py              # FastAPI 入口 (/health, /ask)
│   ├── config.py            # .env 配置读取
│   ├── rag_pipeline.py      # RAG 总流程（待开发）
│   ├── retriever.py         # 混合检索（待开发）
│   ├── answer_policy.py     # 风险分类与拒答（待开发）
│   ├── llm_client.py        # LLM 调用封装（待开发）
│   └── qq_bot.py            # QQ Bot 适配（待开发）
├── scripts/
│   ├── build_chunks.py      # ✅ 文档切分
│   ├── build_index.py       # ✅ 索引构建
│   ├── crawl_sources.py     # 待开发
│   ├── parse_documents.py   # 待开发
│   ├── demo_ask.py          # 待开发
│   └── eval_rag.py          # 待开发
├── data/
│   ├── sources.csv          # ✅ 26 个资料源清单
│   ├── raw/                 # 原始 HTML/PDF（当前跳过）
│   ├── processed/           # 24 个 .md 文件（本地维护，不提交 git）
│   ├── chunks/              # chunks.jsonl（脚本生成，不提交 git）
│   ├── index/               # 向量库/BM25 索引（脚本生成，不提交 git）
│   └── eval/                # 评测问题（待构建）
├── docs/
│   ├── requirement.md       # ✅ MVP 需求文档
│   ├── risk_policy.md       # ✅ 风险策略
│   └── source_priority.md   # ✅ 资料来源优先级
├── tests/                   # 测试（待编写）
├── .env.example             # 环境变量模板
├── requirements.txt         # Python 依赖
└── README.md
```

> **注意**: `data/processed/` 目录中的 `.md` 文件通过其他渠道同步（网盘/群文件），不提交到 Git。

---

## 如何协作

```bash
# 1. 从 master 拉最新
git checkout master && git pull

# 2. 创建功能分支
git checkout -b feat/your-feature-name

# 3. 开发并提交
git add <files>
git commit -m "描述你的改动"

# 4. 推送并在 GitHub 创建 PR
git push -u origin feat/your-feature-name
```

**分工建议：**
- 后端 A：`app/retriever.py` + `app/llm_client.py`
- 后端 B：`app/rag_pipeline.py` + `app/answer_policy.py`
- 资料负责人：维护 `data/sources.csv`，补充文档
- PM：验收回答质量，管理 bad case

---

## 风险提示

本系统仅提供校规和流程的一般性查询，不替代学院教务员、辅导员或相关部门的正式答复。涉及退学、处分、作弊、学位、毕业资格等高风险问题，系统不会直接给出个人结论。
