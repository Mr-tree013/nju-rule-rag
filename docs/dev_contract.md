# 开发契约 v0.1

> A（app/ 在线服务侧）与 B（scripts/data/ 离线数据侧）的共同约定。
> 修改本文件前需两人确认，修改后同步更新两边代码。

---

## 一、chunks.jsonl 字段

B 同学保证每个 chunk 包含以下字段：

```json
{
  "chunk_id": "nju-jw-001-0001",
  "source_id": "nju-jw-001",
  "title": "南京大学普通全日制本科生学籍管理细则",
  "url": "",
  "department": "本科生院",
  "scope": "本科生",
  "priority": 1,
  "section": "第三条",
  "content": "第三条 ……",
  "fetched_at": "2026-05-24 02:16:00"
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `chunk_id` | 是 | `{source_id}-{序号4位}`，稳定可复现 |
| `source_id` | 是 | 对应 `sources.csv` |
| `title` | 是 | 来源标题 |
| `url` | 是 | 可为空字符串（本地文件） |
| `department` | 是 | 发布部门 |
| `scope` | 是 | 适用范围 |
| `priority` | 是 | 1–5，越小越权威 |
| `section` | 是 | 条款号或小标题 |
| `content` | 是 | chunk 正文 |
| `fetched_at` | 是 | 构建时间 `YYYY-MM-DD HH:MM:SS` |

约定：
- A 只能依赖上述字段
- B 可新增字段，不可删除或改名
- 字段变更先改本文档，再同步两边代码
- `validate_chunks.py` 通过后才能交给 A 使用

---

## 二、`POST /ask` 返回格式

A 同学保证返回以下固定结构：

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
      "title": "……",
      "url": "",
      "priority": 1
    }
  ],
  "debug": {
    "retrieval_count": 5,
    "latency": 2.31
  }
}
```

约定：
- B 的 `eval_rag.py` 只调用 `/ask`
- A 的 `qq_bot.py` 只调用 `/ask`
- `sources` 只能来自检索到的 chunks
- 找不到依据时 `sources` 可为空数组，`answer` 必须说明依据不足
- 异常时返回友好错误，不暴露堆栈

---

## 三、index manifest 格式

B 同学构建索引后生成 `data/index/manifest.json`：

```json
{
  "built_at": "2026-05-24 20:00:00",
  "chunks_file": "data/chunks/chunks.jsonl",
  "chunk_count": 396,
  "bm25_index": "data/index/bm25.pkl",
  "chunk_lookup": "data/index/chunk_lookup.json",
  "vector_index": "data/index/chroma",
  "embedding_model": "all-MiniLM-L6-v2",
  "status": "ok"
}
```

约定：
- `app/retriever.py` 启动时读取 `manifest.json`
- 向量索引不可用时优雅降级为纯 BM25
- `build_index.py` 不破坏已有可用索引

---

## 四、Git 分支规则

```
main                      稳定可演示版本
dev                       日常联调分支
feature/app-*             A 同学功能分支
feature/data-*            B 同学功能分支
```

规则：
- 不直接推 main
- 每天至少合并一次到 dev
- main 只放稳定版本
- 合并前跑 `validate_sources.py` + `validate_chunks.py`

---

## 五、每日联调

- **时间**：每晚 21:00（可调整）
- **检查项**：
  ```bash
  python scripts/validate_sources.py
  python scripts/validate_chunks.py
  python scripts/build_index.py
  uvicorn app.main:app --reload
  curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"缓考怎么申请？"}'
  ```

---

## 变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-05-24 | v0.1 | 初始版本 |
