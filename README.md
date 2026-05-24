# NJU Rule RAG

南京大学本科新生校规与教务流程 RAG 问答系统。

基于公开校规资料，支持自然语言提问、来源引用、风险分级与拒答机制。

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

```bash
# 测试
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"缓考怎么申请？"}'
```

---

## 模块状态

### A 同学：app/ 在线服务侧（全部完成）

| 文件 | 功能 |
|------|------|
| `app/main.py` | FastAPI 入口，`/health` `​/ask`，空问题 400，异常友好错误 |
| `app/config.py` | `.env` 配置读取，阈值可配 |
| `app/answer_policy.py` | 风险分类（19 个关键词）、拒答、高风险提示 |
| `app/retriever.py` | BM25 + Chroma 混合检索，manifest 启动加载，权重 0.45/0.35/0.20 |
| `app/llm_client.py` | LLM + Embedding API 封装，自动重试，API Key 脱敏 |
| `app/rag_pipeline.py` | 完整 RAG 流程：分类→检索→过滤→LLM→来源→风控 |
| `app/qq_bot.py` | QQ Bot 适配，`/问` 前缀，【结论】【依据】【提醒】三栏 |

### B 同学：scripts/ + data/ 离线数据侧（全部完成）

| 文件 | 功能 |
|------|------|
| `data/sources.csv` | 26 个资料来源，priority 1-5 |
| `data/processed/` | 24 个 .md 校规文档 |
| `scripts/build_chunks.py` | 按条款切分，长段拆分，短段合并，噪声过滤 → 421 chunks |
| `scripts/build_index.py` | BM25 + Chroma 双索引，ENABLE_VECTOR 可控 |
| `scripts/validate_sources.py` | 校验 sources.csv 字段/唯一性/取值范围 |
| `scripts/validate_chunks.py` | 校验 chunks.jsonl 字段/内容/唯一性 |
| `scripts/crawl_sources.py` | 网页/PDF 抓取，礼貌延迟，跳过需登录源 |
| `scripts/eval_rag.py` | 批量评测 /ask，输出 results.csv + summary.json |
| `data/eval/questions.csv` | 55 道评测问题，10 主题覆盖 |
| `data/chunks/chunk_stats.json` | 每次构建自动生成 |

### 文档

| 文件 | 内容 |
|------|------|
| `docs/requirement.md` | MVP 需求规格 |
| `docs/risk_policy.md` | 风险分级与回答策略 |
| `docs/source_priority.md` | 资料来源优先级 |
| `docs/dev_contract.md` | 开发契约（chunks 字段、/ask 格式、manifest、Git 规则） |
| `docs/evaluation_report.md` | 55 题评测报告 |
| `docs/demo_script.md` | 10 问演示脚本 |

---

## Demo 10 问结果

| # | 问题 | risk | 来源 | 行为 |
|---|------|------|------|------|
| 1 | 缓考在哪里申请？ | medium | 5 | 返回具体流程 |
| 2 | 补考和重修有什么区别？ | medium | 5 | 对比差异 |
| 3 | 选课人数满了怎么办？ | medium | 5 | 拒答（无依据） |
| 4 | 成绩有误应该怎么办？ | medium | 5 | 流程指引 |
| 5 | 学业预警是什么？ | medium | 5 | 定义说明 |
| 6 | 辅修需要注意什么？ | medium | 5 | 注意事项 |
| 7 | 交换课程如何认定？ | medium | 5 | 详细流程 |
| 8 | 我作弊了会不会被开除？ | **high** | 5 | 描述规定不下结论 |
| 9 | 我这种情况还能不能毕业？ | **high** | 5 | 拒答+提醒 |
| 10 | 校历在哪里看？ | low | 5 | 拒答（无依据） |

**10/10 通过，满足 Demo 交付标准。**

---

## 数据流水线

```
data/processed/*.md           ← 清洗后的校规文档（24 个）
        │
  build_chunks.py             ← 按第X条/一、/（一）切分 → 421 chunks
        │
  build_index.py              ← BM25 关键词索引 + Chroma 向量索引
        │
  data/index/
  ├── bm25.pkl                ← BM25 索引 + chunk 数据
  ├── chunk_lookup.json       ← chunk_id → chunk 映射
  ├── manifest.json           ← 索引元信息
  └── chroma/                 ← 向量存储
```

```bash
# 文档更新后重建
python scripts/build_chunks.py
python scripts/build_index.py
python scripts/validate_chunks.py
```

---

## RAG 问答流程

```
用户提问
  │
  ▼
classify_question()     → risk_level: low / medium / high
  │
  ▼
HybridRetriever.search() → BM25(0.45) + Vector(0.35) + Priority(0.20)
  │
  ▼
MIN_RELIABLE_SCORE 过滤  → 低于阈值 → 拒答
  │
  ▼
build_prompt()          → system + context + 高风险追加提醒
  │
  ▼
chat()                  → LLM 生成回答
  │
  ▼
后处理                   → 长度截断(600) + 高风险追加提醒 + 提取来源
  │
  ▼
返回 JSON { question, answer, risk_level, need_human_confirm, sources, debug }
```

---

## 配置说明

复制 `.env.example` 为 `.env`，填入：

```bash
# LLM API（必填，OpenAI 兼容接口即可）
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# 检索参数（可选）
BM25_TOP_K=10
VECTOR_TOP_K=10
HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2
HIGH_RISK_MIN_SCORE=0.25

# 构建选项
ENABLE_VECTOR=true
```

---

## 已知限制

1. **向量中文差**：Chroma ONNX all-MiniLM-L6-v2 对中文无效，建议换 bge-m3
2. **关键词误报**：含「学位」的无关问题偶尔被判为 high
3. **长 chunk 残留**：学生手册 25 段 >800 中文字符，为连续表格无段落边界
4. **DeepSeek API 偶发超时**：已加 3 次重试，但偶有 60s 等待

---

## 下一步

| 优先级 | 任务 | 负责 |
|--------|------|------|
| P0 | 换中文 embedding 模型（bge-m3 或 text2vec） | B |
| P0 | 补全 nju-jw-022/023 两个 HTML 来源的 URL 和抓取 | B |
| P0 | QQ Bot 接入实际 QQ 框架并测试 | A |
| P1 | 优化检索融合参数（根据评测数据调权重） | A |
| P1 | 增加更多高风险测试 case | B |
| P2 | 对接学校官方 RSS/公告源实现自动更新 | B |
| P2 | 部署到服务器供小范围试用 | 共同 |

---

## 风险提示

本系统仅提供校规和流程的一般性查询，不替代学院教务员、辅导员或相关部门的正式答复。涉及退学、处分、作弊、学位、毕业资格等高风险问题，系统不会直接给出个人结论。

> A 负责「服务稳定、回答可信、风险可控、Bot 可用」；B 负责「资料可靠、chunk 可用、索引可复现、评测能闭环」。两人通过 `chunks.jsonl`、`manifest.json`、`/ask` 三个契约协作。
