# NJU Rule RAG

南京大学本科新生校规与教务流程 RAG（检索增强生成）问答系统。

## 功能特性

- 自然语言查询校规、培养方案、办事流程
- 每个回答附带来源引用
- 风险分级回答策略（低/中/高风险）
- 找不到依据时明确拒答
- 支持本地命令行和 FastAPI 接口

## 技术架构

```
用户提问 → 混合检索(BM25+向量) → LLM生成答案 → 风险分级 → 返回{答案, 来源, 风险等级}
```

- **后端框架**: FastAPI + Uvicorn
- **检索**: BM25 (rank-bm25) + 向量检索 (Chroma)
- **LLM**: 可替换接口 (OpenAI / DeepSeek / 通义千问 / 智谱)
- **数据处理**: BeautifulSoup + PyMuPDF + pdfplumber

## 目录结构

```
nju-rule-rag/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置读取
│   ├── rag_pipeline.py      # RAG 总流程
│   ├── retriever.py         # 检索逻辑 (BM25 + 向量 + 混合)
│   ├── answer_policy.py     # 风险分类与拒答策略
│   ├── llm_client.py        # LLM 调用封装
│   └── qq_bot.py            # QQ Bot 适配层
├── scripts/
│   ├── crawl_sources.py     # 抓取网页/PDF
│   ├── parse_documents.py   # 解析 HTML/PDF
│   ├── build_chunks.py      # 文档切分
│   ├── build_index.py       # 构建索引
│   ├── demo_ask.py          # 本地命令行问答
│   └── eval_rag.py          # 评测脚本
├── data/
│   ├── sources.csv          # 资料源清单
│   ├── raw/                 # 原始 HTML/PDF
│   ├── processed/           # 清洗后文本
│   ├── chunks/              # chunks.jsonl
│   ├── index/               # 向量库/BM25 索引
│   └── eval/                # 评测问题
├── docs/
│   ├── requirement.md       # MVP 需求文档
│   ├── risk_policy.md       # 风险策略
│   └── source_priority.md   # 资料来源优先级
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## 本地运行

### 环境要求

- Python 3.10+
- pip

### 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填入 LLM API key 等信息
```

### 启动

```bash
uvicorn app.main:app --reload
```

访问 http://127.0.0.1:8000/docs 查看 Swagger UI。

### 测试接口

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 提问
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "缓考怎么申请？"}'
```

## 更新资料库

1. 编辑 `data/sources.csv` 添加/修改资料源
2. 运行 `python scripts/crawl_sources.py` 下载原始文件
3. 运行 `python scripts/parse_documents.py` 解析为文本
4. 运行 `python scripts/build_chunks.py` 切分为 chunks
5. 运行 `python scripts/build_index.py` 构建检索索引

## 运行评测

```bash
python scripts/eval_rag.py
```

结果输出到 `data/eval/results.csv`。

## 当前限制

- 知识库覆盖范围有限（初期 20-40 个资料源）
- 不处理需要登录认证的页面
- 不对个人情况做判断（退学、处分、学位等）
- 回答仅基于已入库资料，资料更新频率取决于人工维护

## 风险提示

本系统仅提供校规和流程的一般性查询，不替代学院教务员、辅导员或相关部门的正式答复。涉及退学、处分、作弊、学位、毕业资格等高风险问题，系统不会直接给出个人结论，请务必以官方书面答复为准。
