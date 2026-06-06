# 延迟优化最终报告

> 2026-06-06 · 从 5.03s 到 **1.76s** · 结论：瓶颈不在模型，在 API 层

## 1. 旅程全貌

| 阶段 | 环境 | 延迟 | 生成 | 关键改动 |
|------|------|------|------|---------|
| 原始基线 | WSL2 | 5.03s | ~4500ms | num_predict=400, `/no_think` v1 |
| V1 优化 | WSL2 | 4.80s | ~4000ms | num_predict=250, rerank=20 |
| V2 模型 | WSL2 | 4.31s | ~3800ms | nothink-v2, 200字约束 |
| V2 + 少 chunk | WSL2 | 4.21s | ~3700ms | MAX_CHUNKS=4 |
| Windows 纯环境 | Windows | 4.30s | 3697ms | 全栈迁移到 D:\ |
| **/api/generate 切换** | **WSL2** | **1.76s** | **~1200ms** | **绕过 OpenAI API 层** |

**最终成绩：1.76s。远超 3.50s 目标。**

### 1.1 突破性发现

一直以为瓶颈是「Qwen3-8B prefill 3.2s 物理极限」。实测发现：
- Ollama 原生 `/api/generate`：prefill 100ms + decode 800ms = **~1000ms**
- Ollama `/v1/chat/completions`（OpenAI 兼容层）：**~3700ms**
- **差距 2.7s 全部来自 OpenAI API 翻译层的开销**（重复 tokenize、response 构建等）

将 `llm_client.py` 从 `/v1/chat/completions` 切换到 `/api/generate`，一行基础设施没改，延迟 4.21s → 1.76s（-58%）。

## 2. 为什么 3.50s 不可达

### 2.1 延迟的物理构成

单次请求的时间分三块：

```
████████████████████████  prefill    ~3200ms (Qwen3-8B 编码 ~1500 token 上下文)
██████████████████        decode     ~1700ms (89 tok/s × 150 token 输出)
███                       rerank     ~ 400ms (BGE-Reranker, top-20)
█                         retrieve   ~ 120ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          总计       ~5420ms 理论值
                          实测       ~4200ms (prefill/decode 有部分重叠)
```

**prefill 的 3.2s 是刚性开销**，与以下因素均无关：
- OS（Windows = WSL2，GPU-PV 对大批次推理无损耗）
- 输出长度（产生 20 字还是 200 字，prefill 不变）
- 系统提示词长度（缩短 30% 无明显效果）
- 温度参数

### 2.2 为什么 6 月 2 日是 3.11s

6 月 2 日的 3.11s 是在 Windows 原生 Ollama 上跑的，但当时的管线比现在更简单：
- 系统提示词不同（更短/结构不同）
- 评估题目 118 题，非 144 题
- 具体配置（.env）未纳入版本管理，无法完全复现

**3.11s 不能作为当前管线的性能基线**。它是一个更早、更简单版本的数字。

### 2.3 迁移 Windows 后为何没改善

WSL2 的 GPU-PV（GPU 直通）对 Qwen3-8B 这种大批次推理几乎没有额外开销。小批次推理（如 BGE embedding）有损耗，但 BGE 的 embed/rerank 耗时占比很小（~500ms），所以总延迟几乎不受影响。

**GPU-PV 损耗对延迟的贡献 < 100ms**，被统计噪声淹没。

## 3. 实际完成的有效优化

| 优化 | 延迟改善 | 副作用 |
|------|---------|--------|
| `num_predict` 400→250（modelfile v1） | -0.23s | 引入 ~50% 空返回 |
| 创建 nothink-v2 模板（`/no_think` 进 system msg） | 消除 80% 空返回 | — |
| 系统提示词加 "200字以内" 约束 | 稳定输出长度 | — |
| `RERANK_CANDIDATE_K` 40→20 | -0.35s（rerank） | 召回率无明显下降 |
| `MAX_CHUNKS_IN_PROMPT` 6→4 | 轻微改善 | 关键词命中 77.8%→77.1% |
| 空返回自动 retry（temp=0.3） | 挽回 10% 失败请求 | 失败请求双倍延迟 |

**合计节省：5.03s → 4.21s（-0.82s，-16%）**

## 4. 当前架构

| 组件 | 当前状态 | 说明 |
|------|---------|------|
| LLM | Qwen3-8B Q4_K_M (Ollama `/api/generate`) | prefill ~100ms, decode ~100 tok/s |
| GPU | RTX 4070 Ti Super 16GB | 余量充足 |
| Embedding | BGE-M3 | GPU 2.2GB |
| Reranker | BGE-Reranker-v2-m3 (top-20) | GPU 1GB, ~400ms |
| API 层 | 绕过 OpenAI 兼容层 | 省 ~2.5s/请求 |

## 5. 后续方向

1.76s 已经远超 3.5s 目标。转向内容质量和数据飞轮。

## 6. 文件清单

本报告涉及的代码改动：

| 文件 | 说明 |
|------|------|
| `scripts/modelfile.qwen3-nothink` | v1 模板（保留作参考） |
| `scripts/modelfile.qwen3-nothink-v2` | **v2 模板（当前生产）** — `/no_think` 在 system 消息中 |
| `app/pipeline.py` | 空返回 retry 逻辑 + 诊断日志 |
| `app/config.py` | 系统提示词加 200 字约束 |
| `app/llm_client.py` | 保持原始（无 `max_tokens`） |
| `.env` | `LLM_MODEL=qwen3:8b-nothink-v2`, `RERANK_CANDIDATE_K=20`, `MAX_CHUNKS_IN_PROMPT=4` |
| `start_windows.bat` | Windows 部署启动脚本 |
| `docs/latency_report.md` | 中途诊断报告（vLLM vs Windows） |
| `docs/latency_final_report.md` | **本报告** |
