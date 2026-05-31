# Reranker Fine-tuning 实验报告

**实验时间**: 2026-05-31  
**目标**: 提升 RAG pipeline 的 Context Precision（当前 0.12），修复 reranker sigmoid discrimination 弱的问题

---

## 背景

当前 RAG pipeline 使用 `BAAI/bge-reranker-v2-m3` 做 cross-encoder reranking，但存在两个问题：

1. **Context Precision@10 = 0.12**：只有 ~1/8 的检索 chunk 是真正相关的，LLM 没有可靠锚点，导致 Faithfulness 只有 2.31/5
2. **Reranker score separation 极弱**：cross-encoder 输出的 logit 全部聚集在 0 附近（sigmoid ~0.5），几乎没有排序能力。v0.5.2 引入了 `0.4×原始 + 0.6×sigmoid(logit)` 的 score fusion 作为 workaround

## 实验方法

### 训练数据

从 eval 的 gold-source 标注自动构建 relevance pairs：

| 类型 | 数量 |
|------|------|
| 训练集 | 532 pairs |
| 测试集 | 133 pairs |
| 正样本 | query + 正确来源 chunk（label=1） |
| 负样本 | query + 同 topic 的错误 chunk（label=0） |

数据示例：
```
query: "退学之后还有机会回来吗？"
content: "学士学位授予条件与程序等按《南京大学学士学位授予实施办法》中相关规定执行。"
label: 1  (正确来源)

query: "毕业论文没过能毕业吗？"
content: "各院系应根据公布的学科、专业准入年度实施计划和方案..."
label: 0  (不相关，只是同 topic)
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | BAAI/bge-reranker-v2-m3 |
| 架构 | Cross-encoder (XLM-RoBERTa, 568M params) |
| Epochs | 3 |
| Batch size | 4（降至 4 防止 16GB VRAM OOM）|
| Learning rate | 2e-5 |
| Warmup steps | 39 (10%) |
| Optimizer | AdamW |
| Loss | Binary Cross-Entropy |
| 训练时间 | 40 分钟 (RTX 4070 Ti Super 16GB) |

### 遇到的坑

1. **VRAM 溢出导致 WSL 崩溃**：训练时 Ollama 仍加载 Qwen3-8B (~5.2GB)，加上训练显存直接打满 16GB → GPU 驱动崩溃 → WSL 重启。解决：训练前卸载 Ollama 模型，BATCH_SIZE 从 8 降到 4
2. **模型未落盘**：sentence-transformers v5.5.1 的 `CrossEncoder.fit(save_best_model=False)` 在无 evaluator 时不会保存模型。`SaveModelCallback` 只在 `on_evaluate` 回调中保存（需要 evaluator + save_best_model=True）。修复：`fit()` 后手动 `model.save()`

---

## 实验结果

### 训练指标（在 133 条测试集上）

| 指标 | 训练前 | 训练后 | 变化 |
|------|--------|--------|------|
| Positive score mean | 0.17 | 0.88 | +0.71 |
| Negative score mean | 0.008 | 0.22 | +0.21 |
| **Score separation** | **0.16** | **0.66** | **+4x** |
| Train loss | - | 0.29 | 正常收敛 |

Score separation 提升显著——模型在领域测试集上学会了有效区分相关/不相关 chunk。

### 检索评估（118 题，带 rerank）

| 指标 | 原始 Reranker | Fine-tuned | 变化 |
|------|-------------|------------|------|
| recall@5 | **0.881** | 0.839 | **-0.042** |
| MRR | **0.612** | 0.516 | **-0.096** |
| Context Precision@10 | 0.12 | 0.12 | 0 |

**Fine-tuned 模型在真实检索任务上全面退步。**

---

## 分析：为什么会退步？

**核心原因：Catastrophic Forgetting**

原始 BGE-Reranker-v2-m3 在百万级多语言 cross-encoder 数据上训练，具备通用的 query-document 相关性判断能力。我们仅用 532 条 NJU 领域数据 fine-tune：

- 训练数据太小（532 vs 百万级），模型迅速 overfit 到领域数据的浅层模式
- 3 epochs × 532 samples = 1596 次更新，足以让模型遗忘预训练权重中的通用排序知识
- 模型学会了区分"同一个 topic 下的相关/不相关 chunk"（测试集上 score separation 很好），但在真实检索场景中，候选 chunk 来自不同 topic，分布与训练数据不同，泛化失败

**次要因素**：

- 训练/测试数据同分布（都从 gold-source 构造），测试集的 score separation 提升无法代表真实泛化能力
- 负样本构造方式（同 topic 随机负样本）过于简单，模型可能学到了简单的 shortcut 而非真正的语义匹配

---

## 待讨论的问题

1. **数据量门槛**：用 sentence-transformers fine-tune BGE-Reranker-v2-m3 做领域适配，大概需要多少训练对才能避免灾难性遗忘？有没有经验法则（如 5K/10K/50K）？
2. **训练策略**：小数据集场景下，以下哪种更有效：
   - 极低 LR（1e-6）+ 1 epoch 的轻量微调？
   - LoRA / adapter 只更新部分参数？
   - 混合训练（原始数据回放 + 领域数据）？
   - 直接用 prompt-based reranking（让 LLM 做 pairwise 比较）替代 cross-encoder？
3. **替代方案**：考虑到我们只有 532 对训练数据，是否应该放弃 reranker fine-tuning，转而投入其他方向？比如：
   - 用 LLM-as-reranker（listwise/pairwise）——我们有 Qwen3-8B 本地部署
   - 改进 hybrid retrieval 权重（目前 BM25 0.25 / Vector 0.45 / Priority 0.30）
   - 增加更多源文档来提升 recall 上限
4. **评估问题**：我们的 eval 用 `pred > 0` 作为二分类阈值，但 fine-tuned 模型输出分布已变（pos ~0.88, neg ~0.22），需要重新校准阈值。不过 retrieval ranking 用的是相对排序而非绝对阈值，所以不影响 recall/MRR 的结论。

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/finetune_reranker.py` | 训练脚本（已修复 save 问题） |
| `data/models/bge-reranker-nju/` | Fine-tuned 模型 (2.2 GB) |
| `data/eval/reranker_train.jsonl` | 训练数据 (532 pairs) |
| `data/eval/reranker_test.jsonl` | 测试数据 (133 pairs) |
| `data/eval/retrieval_summary_rerank.json` | Retrieval eval 结果 |

---

## 环境信息

- GPU: RTX 4070 Ti Super 16GB
- CUDA: 12.4
- sentence-transformers: 5.5.1
- BGE-Reranker-v2-m3: 568M params, XLM-RoBERTa 架构
- LLM: Qwen3-8B (Ollama, no-think mode)
