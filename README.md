# NJU Rule RAG

南京大学本科校规与教务流程 RAG（检索增强生成）问答系统。

基于 70 份校规、办事指南和校园生活文档，支持自然语言提问、来源引用、风险分级与拒答机制。已接入 QQ Bot（NapCat + OneBot v11）。

**当前状态**：v0.3.0，全流程可运行，107 个测试通过，3227 个可检索片段。

---

## 部署方式

### 方式一：本地部署

```bash
git clone https://github.com/Mr-tree013/nju-rule-rag.git
cd nju-rule-rag
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                   # 编辑填入 LLM_API_KEY
uvicorn app.main:app --port 8000 --reload
```

> 首次运行会加载中文 embedding 模型（~1.9GB），启动需约 30 秒。模型已预缓存于 `data/models/`，无需联网下载。

### 方式二：Docker 部署

```bash
cp .env.example .env   # 编辑填入 API Key
docker compose up -d
```

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
sources.csv (70条来源) → processed/*.md → build_chunks.py → chunks.jsonl (3227条)
                                                                    │
                                          build_index.py ────────────┘
                                          BM25 (jieba) + Chroma (text2vec-base-chinese)
```

### 更新索引（添加新文档后）

```bash
python scripts/build_chunks.py
python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py
```

### 文档格式转换

```bash
python scripts/parse_to_markdown.py input.html --title "标题" -o data/processed/输出.md
python scripts/parse_to_markdown.py input.pdf  --title "标题" -o data/processed/输出.md
```

---

## 配置参考

```bash
# 必填
LLM_API_KEY=sk-your-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# QQ Bot
QQ_BOT_SELF_ID=你的机器人QQ号

# 检索调参（可选）
BM25_TOP_K=10
VECTOR_TOP_K=10
HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2
HIGH_RISK_MIN_SCORE=0.25

# 向量检索（可选，默认启用）
ENABLE_VECTOR=true
LOCAL_EMBEDDING_MODEL=shibing624/text2vec-base-chinese
```

支持所有 OpenAI 兼容接口（DeepSeek / 通义千问 / 智谱 / OpenAI）。

---

## 在线端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ask` | POST | 问答接口 `{"question": "..."}` |
| `/qq` | POST | QQ Bot webhook（NapCat 回调） |

---

## 评测

```bash
uvicorn app.main:app --port 8000 &
python scripts/eval_rag.py          # 70 题评测
python -m json.tool data/eval/summary.json
```

---

## 资料库概况

| 类别 | 数量 |
|------|------|
| 来源总数 | 70 |
| 本科生院文档 | 47 |
| 南哪助手生活指南 | 22 |
| 南京大学 | 1 |
| 可检索片段 | 3227 |
| 评测问题 | 70 题 |

---

## 后续方向

### 优化

- **响应速度**：当前 DeepSeek API 为最大延迟瓶颈，评估切换到更快的模型或本地 LLM
- **混合检索权重**：当前 BM25 (0.45) / Vector (0.35) / Priority (0.20)，可通过评测数据调优
- **嵌入模型升级**：`text2vec-base-chinese` → `bge-m3` 提升语义检索质量
- **增量索引**：文档更新时只重建变更部分，避免全量重跑

### 测试

- **风险分类准确率**：关键词分类器有误报风险（如"学位证"误判 medium），需统计误报率并考虑升级为 LLM 分类
- **拒答边界**：验证"无相关资料" vs "不确定"的边界判断是否合理
- **来源引用正确性**：抽查 LLM 回答引用的 chunk 是否真实对应
- **端到端回归**：70 题评测需要配置 API Key 后定期重跑，跟踪回答质量变化
- **压力测试**：多群并发 QQ Bot 请求时的稳定性

---

> 本系统仅提供一般性校规和生活信息查询，不替代教务员或辅导员的正式答复。高风险问题不会给出个人结论。
