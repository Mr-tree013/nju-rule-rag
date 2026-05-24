# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 31 份公开校规和办事指南，支持自然语言提问、来源引用、风险分级与拒答机制。

**当前状态**：Day 1-6 全部完成，端到端可运行，Demo 10 问 10/10 通过。

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

---

## 项目结构

```
nju-rule-rag/
├── app/                          # 在线问答服务
│   ├── main.py                   # FastAPI 入口
│   ├── config.py                 # 配置读取（.env）
│   ├── answer_policy.py          # 风险分类 + 拒答
│   ├── retriever.py              # BM25 + Chroma 混合检索
│   ├── llm_client.py             # LLM API 封装（重试/脱敏）
│   ├── rag_pipeline.py           # RAG 全流程编排
│   └── qq_bot.py                 # QQ Bot 适配层
│
├── scripts/                      # 离线数据处理
│   ├── build_chunks.py           # MD → 435 chunks（条款切分）
│   ├── build_index.py            # chunks → BM25 + Chroma 索引
│   ├── validate_sources.py       # 校验 sources.csv
│   ├── validate_chunks.py        # 校验 chunks.jsonl
│   ├── crawl_sources.py          # 从 URL 抓取原始文件
│   ├── parse_documents.py        # 批量解析（sources.csv 驱动）
│   ├── parse_to_markdown.py      # ★ 通用格式转换器
│   └── eval_rag.py               # 批量评测 /ask
│
├── data/
│   ├── sources.csv               # 31 条资料来源清单
│   ├── processed/                # 31 个 .md 校规文档
│   ├── chunks/                   # chunks.jsonl（435条）+ stats
│   ├── index/                    # BM25 + Chroma + manifest
│   ├── eval/                     # 55 题评测集 + 结果
│   └── raw/                      # 原始抓取文件
│
├── docs/
│   ├── requirement.md            # MVP 需求文档
│   ├── risk_policy.md            # 风险分级策略
│   ├── source_priority.md        # 资料来源优先级
│   ├── dev_contract.md           # 开发契约（字段/接口/规则）
│   ├── evaluation_report.md      # 55 题评测报告
│   └── demo_script.md            # Demo 10 问脚本
│
├── tests/
│   └── test_answer_policy.py     # 26 个单元测试
│
├── .env.example                  # 环境变量模板
├── requirements.txt              # 13 个 Python 依赖
└── README.md
```

---

## 模块状态

### app/ — 全部完成

| 文件 | 功能 |
|------|------|
| `main.py` | `/health` `​/ask`，空问题 400，异常降级友好错误 |
| `config.py` | 所有阈值/路径/模型名从 .env 读取 |
| `answer_policy.py` | 20+ 高风险关键词，三级分类，拒答模板 |
| `retriever.py` | 双路混合检索，权重 0.45/0.35/0.20，启动读 manifest |
| `llm_client.py` | OpenAI 兼容接口，3 次重试，Key 脱敏 |
| `rag_pipeline.py` | 分类→检索→过滤→LLM→来源→风控，全流程 |
| `qq_bot.py` | `/问` 触发，【结论】【依据】【提醒】三栏，≤800 字 |

### scripts/ — 全部完成

| 文件 | 功能 |
|------|------|
| `build_chunks.py` | 第X条/一、/（一）/1.2.3. 切分，800cn 长分，30cn 短合，噪声过滤 |
| `build_index.py` | jieba BM25 + text2vec-base-chinese Chroma，ENABLE_VECTOR 可控 |
| `validate_sources.py` | 必要字段/唯一性/取值范围，exit 1 on error |
| `validate_chunks.py` | 字段/内容/唯一性，exit 1 on error |
| `crawl_sources.py` | 礼貌延迟 1s，跳过 need_login，失败不中断 |
| `parse_documents.py` | 批量解析：PDF(PyMuPDF) + DOC(LibreOffice) |
| `parse_to_markdown.py` | ★ 通用转换器：HTML/PDF/DOC/DOCX/TXT → MD |
| `eval_rag.py` | 55 题批量评测，输出 results.csv + summary.json |

### data/ — 全部就绪

| 内容 | 数量 |
|------|------|
| 资料来源 | 31 条（priority 1: 13, 2: 9, 3: 2, 4: 5, 5: 2） |
| Markdown 文档 | 31 个（24 校规 + 2 考试通知 + 5 操作手册/论文规范） |
| 可检索片段 | 435 chunks |
| 评测问题 | 55 题 / 10 主题 |

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
│  data/processed/*.md  ←── 31 个清洗后文档          │
│                                                    │
└───────────────────┬────────────────────────────────┘
                    │
┌─ 索引构建 ──────────────────────────────────────┐
│                                                    │
│  build_chunks.py                                   │
│  按条款切分 → 长段拆分 → 短段合并 → 噪声过滤       │
│       │                                            │
│       ▼                                            │
│  chunks.jsonl (435 chunks) + chunk_stats.json       │
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
│  classify_question()  →  risk: low/medium/high     │
│       │                                            │
│       ▼                                            │
│  HybridRetriever: BM25(0.45)+Vector(0.35)+Pri(0.20)│
│       │                                            │
│       ▼                                            │
│  分数过滤 → 低于阈值 → 拒答                         │
│       │                                            │
│       ▼                                            │
│  LLM 生成 → 长度截断 → 高风险追加提醒 → 提取来源   │
│       │                                            │
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

在代码中调用：

```python
from scripts.parse_to_markdown import convert_file
from pathlib import Path

md = convert_file(Path("通知.html"), title="考试安排", url="https://jw.nju.edu.cn/...")
Path("data/processed/考试安排.md").write_text(md, encoding="utf-8")
```

---

## Demo 10 问

| # | 问题 | risk | 行为 |
|---|------|------|------|
| 1 | 缓考在哪里申请？ | medium | 返回在线申请具体流程 |
| 2 | 补考和重修有什么区别？ | medium | 对比条件/成绩/费用 |
| 3 | 选课人数满了怎么办？ | medium | 提示补选阶段即选即中 |
| 4 | 成绩有误应该怎么办？ | medium | 查分→更正流程 |
| 5 | 学业预警是什么？ | medium | 定义 + 帮扶措施 |
| 6 | 辅修需要注意什么？ | medium | 学籍不变/主修合格 |
| 7 | 交换课程如何认定？ | medium | 备案→材料→审核→复审 |
| 8 | 我作弊了会不会被开除？ | **high** | 描述处分规定，不下结论 |
| 9 | 我这种情况还能不能毕业？ | **high** | 拒答 + 提醒咨询教务员 |
| 10 | 校历在哪里看？ | low | 拒答（无依据） |

**10/10 通过**，详见 `docs/demo_script.md`。

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

# 跑 55 题评测
python scripts/eval_rag.py

# 查看汇总
python -m json.tool data/eval/summary.json
```

详细报告见 `docs/evaluation_report.md`。

---

## 技术栈

| 层 | 技术 |
|----|------|
| 接口 | FastAPI + Uvicorn |
| 关键词检索 | BM25 (rank-bm25) + jieba |
| 语义检索 | ChromaDB + text2vec-base-chinese |
| 文本生成 | OpenAI 兼容 API |
| 风控 | answer_policy.py（关键词+规则） |
| 格式转换 | PyMuPDF + BeautifulSoup + LibreOffice |

---

## 已知限制

1. 关键词误报：含「学位」的非高风险问题偶尔被判为 high
2. 学生手册 25 段 >800 中文字符（连续表格，无段落边界可切）
3. DeepSeek API 偶发超时（已加 3 次重试）
4. 3 份教务系统手册以截图为主，文字量少

---

## 下一步

| 优先级 | 任务 |
|--------|------|
| P0 | QQ Bot 接入实际框架测试 |
| P0 | 通知频道自动抓取 + 增量索引 |
| P1 | 优化融合权重 + 扩充高风险 case |
| P2 | 部署供小范围试用 |

---

> 本系统仅提供一般性校规查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
>
> A（在线服务）× B（离线数据）× 契约（chunks.jsonl / manifest.json / POST /ask）
