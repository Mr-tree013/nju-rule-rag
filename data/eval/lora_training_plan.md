# LoRA Fine-tune Qwen3-8B — 训练与评估方案

> 目标：用 4800+ 对 NJU 领域 QA 训练数据，LoRA 微调 Qwen3-8B，在 16GB VRAM 下完成，并安全评估效果。

---

## 一、为什么要做

Phase 0 诊断结论：65% 的 F≤2 失败是 **Type B+C（LLM 拿到了正确文档但仍然编造/忽略）**。Prompt 工程推了 3 轮，F 从 2.31 到 2.70 就到天花板了。LoRA 训练模型"只用资料回答"的行为模式，预期突破这个天花板。

## 二、训练数据

| 来源 | 数量 | 格式 |
|------|------|------|
| LLM 合成（5 轮） | ~1400 | query + positive_chunk_id + hard_negative_chunk_ids |
| 回译增强（5 轮） | ~2500 | 同上 |
| Eval 挖掘 | 195 | 同上 |
| 已有 finetune pairs | 665 | query + content + label |
| **去重后总计** | **~4800** | `data/training/all_pairs_filtered.jsonl` |

每个 pair 包含：
- `query`: 学生问题
- `positive_chunk_id`: 正确答案所在的 chunk ID
- `hard_negative_chunk_ids`: 从真实 BM25+vector top-40 中挖出的 5 个干扰 chunk
- `source_id`: 来源文档

## 三、已有的基础

```bash
# 环境
CUDA 12.4, PyTorch 2.6.0+cu124
GPU: RTX 4070 Ti Super 16GB
Python 3.12

# 已安装
transformers 5.9.0, accelerate 1.13.0, peft (刚装)
bitsandbytes (刚装)

# 模型
Qwen3-8B 已在 Ollama 运行（qwen3:8b-nothink）
BGE-M3 + BGE-Reranker-v2-m3 通过 sentence-transformers 加载

# 训练数据
data/training/all_pairs_filtered.jsonl — 4158 对（+665 旧格式）
data/chunks/chunks.jsonl — 3248 chunks
```

## 四、待解决的 4 个问题

### Q1: 模型加载策略

Qwen3-8B 在 HuggingFace 是 `Qwen/Qwen3-8B`（约 16GB FP16）。16GB VRAM 放不下完整模型 + 训练状态。有两个方案：

**方案 A: QLoRA（4-bit 量化 + LoRA）**
- 模型加载为 4-bit，占用 ~6-8GB VRAM
- LoRA 适配器参数极少（~1%），训练峰值 ~12-14GB
- 使用 `bitsandbytes` + `peft`
- 优点：确定能跑在 16GB 上
- 缺点：4-bit 量化可能损失一些精度

**方案 B: Unsloth（优化的 QLoRA）**
- Unsloth 专门优化了 Qwen3 系列的 LoRA 训练
- 比标准 QLoRA 快 2-3x，显存更省
- 需要 `pip install unsloth`
- 优点：更快更省，社区推荐

**你倾向哪个？** 方案 B 更稳，但需要额外安装。方案 A 依赖已安装的库。

### Q2: 训练数据格式

当前数据是 query + chunk_id，需要转化为模型可训练的格式。两种方案：

**方案 A: 指令微调格式（推荐）**
```
<|im_start|>system
你是南大学长，根据参考资料回答问题。只用资料中的信息，不要编造。<|im_end|>
<|im_start|>user
参考资料：[chunk内容]
问题：[query]<|im_end|>
<|im_start|>assistant
[从chunk内容中提取的答案]
```

- 需要：对每个 pair 生成一个"标准答案"（从 positive chunk 提取）
- 问题：我们没有答案文本，只有 chunk 内容 → 需要用 LLM（DeepSeek 或 Qwen）批量生成答案
- 优点：让模型学会"根据资料生成答案"的完整行为

**方案 B: 对比学习格式（更简单）**
- 直接用 `(query, positive_chunk, [negative_chunks])` 做对比学习
- 使用 sentence-transformers 的 `CrossEncoder.fit()` 或 `MultipleNegativesRankingLoss`
- 优点：不需要生成答案，直接用现有数据
- 缺点：这是训练 retriever/reranker 的方式，不是训练 LLM 生成能力

**你倾向哪个？** 

- 方案 A 直接改善 LLM 生成质量（Type B+C 的根因），但需要先生成答案
- 方案 B 更简单但效果方向不对（改善检索，而我们的瓶颈不是检索）

### Q3: 如何防止灾难性遗忘

上次 532 对全参数微调 BGE-Reranker 直接导致 recall 暴跌。教训：
1. LoRA 只更新 ~1% 参数，天然抗遗忘
2. 但 4800 对全部是 NJU 域数据，可能让模型"忘记"通用中文能力

**建议措施**：
- 混入 20-30% 通用 QA 数据（如 MS MARCO 中文版、alpaca-zh 等）作为回放
- 或：训练 1-2 epoch 就停（不要 3 epoch）
- 或：在训练数据中保留一些"非校规"的通用问答对

**是否需要？** 还是直接用 4800 对 NJU 数据训，靠 LoRA 本身的抗遗忘能力？

### Q4: 如何评估效果

上次 fine-tune reranker 的教训是 **训练集上的 score_separation 是假信号**。这次必须用真实 pipeline 评估：

| 评估方式 | 操作 | 耗时 |
|----------|------|------|
| 离线 eval | 用 `eval_generation.py` 跑 144 题 | ~15min |
| 单题 A/B | 选 5 个已知失败案例，新旧模型对比 | ~2min |
| E2E test | `eval_rag.py` 全量 | ~10min |
| 检索 stability | `eval_retrieval.py` 确认检索不受影响 | ~5min |

**建议流程**：
1. 训练前：记录 baseline（F=2.70, R=3.83）
2. 训练后：用 Ollama 加载新模型（需要转换回 GGUF），跑全套 eval
3. 对比：F 提升 ≥ 0.2 为成功，≤ 0 为失败，0~0.2 为边缘

### Q5: 部署回 Ollama 的流程

LoRA 产出是 PEFT adapter weights（~100MB），不是完整模型。部署到 Ollama 需要：

1. 将 LoRA adapter merge 回 base model → 得到完整 PyTorch 权重
2. 用 `llama.cpp` 的 `convert.py` 转成 GGUF 格式
3. `ollama create` 注册新模型

或：不 merge，用 `vllm` 直接加载 base + LoRA adapter（不需要转 GGUF）

**这一步的具体命令需要验证**。你知道怎么把 PyTorch + LoRA 转成 Ollama 可用格式吗？

## 五、建议的完整流程

```
1. [选择方案] 确定 Q1-Q5 的答案
2. [准备数据] 如果需要方案 A（指令微调），用 DeepSeek 批量生成 4800 个答案
3. [训练] 跑 LoRA（1-2 epoch, ~1-2 小时）
4. [评估] A/B 对比 5 个已知失败案例 → 全量 eval
5. [部署] 转 GGUF → Ollama 注册 → 切换
6. [监控] 跑一周看线上反馈
```

## 六、风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| VRAM OOM | 低 | 训练中断 | 4-bit QLoRA 已确认 16GB 可跑 |
| 灾难性遗忘 | 中 | 通用能力下降 | LoRA rank=8 + 少 epoch |
| 训练不收敛 | 低 | 浪费 GPU 时间 | 先跑 1 epoch test mode |
| eval 测不准 | 高 | 误判效果 | 必须用 pipeline eval，不看 training loss |
| GGUF 转换失败 | 中 | 部署不了 | 提前验证转换流程 |
