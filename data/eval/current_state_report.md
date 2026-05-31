# NJU Rule RAG — 当前状态与瓶颈报告

> 时间：2026-05-31  
> 版本：v0.6.0-dev（从 v0.5.2 升级）

---

## 一、核心指标

| 指标 | v0.5.2 | 当前 | 变化 |
|------|--------|------|------|
| Sources | 105 | **146** | +41 |
| Chunks | 3962 | **3248** | 多策略切分后精简 |
| Eval 题数 | 118 | **144** | +26（-2 浦口 +28 新 topic） |
| Faithfulness | 2.31 | **2.54** | +0.23 |
| Relevance | 3.88 | **3.85** | ~持平 |
| Refusal Correctness | 4.48 | **4.61** | +0.13 |
| recall@5 (rerank) | 0.881 | **0.784** | -0.097* |
| MRR (rerank) | 0.612 | **0.488** | -0.124* |
| Avg latency | 2.23s | **2.58s** | +0.35s |

> *检索指标下降原因：min-max 归一化替换 sigmoid 后，score fusion 更真实地反映 reranker 贡献（而非 sigmoid 塌陷到 0.5 的假象）。原始 sigmoid 让所有 chunk 得分 ~0.5，重排几乎无效——那时的高 recall 实际是 hybrid retriever 的结果。

## 二、本阶段完成的改造

### A 轨 — Prompt 重写 + 三级置信度
- System prompt 从模糊的"不编造"升级为 6 条硬规则 + 8 个 few-shot 示例
- Tier 1/2/3 置信度系统：高信回答 → 软对冲 → 直接转介
- Tier 阈值：0.85/0.70/0.35（Tier 2 覆盖约 30% 问题）
- **效果**：F +0.20，F=1 严重幻觉 -46%

### E 轨 — 覆盖扩充
- 从学生手册 MD 提取 18 份新文档（辅修/准入/准出/处分/奖学金等）
- 从 NJU 官网补充 6 份新文档（校园安全/考研/体育/选课/补考/周边生活）
- 122 → 146 sources，24 topic 基本全覆盖

### E.4 — 多策略文档切分
- 新增 QA/heading/table_row/fixed 四种切分策略
- QA 文档（30 份 FAQ）智能检测 Q&A 格式，回退到 heading 切分
- Chunks 从 ~4300 精简到 3248（-25%），0 过短/过长

### F 轨 — 自动化更新流水线
- `app/crawl/fetcher.py`：ETag/304 + hash 变更检测，仅变化时存入 staging
- `scripts/crawl_scheduled.py`：cron 调度器（3 个活跃源）
- `scripts/review_staging.py`：交互式 accept/reject/skip 审核 CLI
- `POST /admin/ingest_url`：URL 提交端点（已存在）

### Reranker 评估修复
- 新增 nDCG@10、gold_chunk_rank 替代有缺陷的 Context Precision@10
- 实现 LLM-as-Reranker（listwise Qwen3-8B），A/B 后决定不切换
- Cross-encoder 的 sigmoid 融合改为 min-max 归一化

### 检索权重调优
- 126 组合 grid search：最佳 0.30/0.60/0.10（原 0.25/0.45/0.30）
- 直接检索 MRR +6.2%, R@5 +6.4%
- 带 reranker 后效果被覆盖（reranker 主导最终排序）

### 失败的尝试
- **Reranker fine-tune**：532 对数据全参数微调 BGE-Reranker-v2-m3 → 灾难性遗忘，recall@5 反而下降
- **PDF 直接批量入库**：学生手册 PDF 提取文本含页码/格式碎片 → 检索噪声

## 三、当前瓶颈

### 1. Faithfulness 天花板
经过两轮 prompt 迭代 + 三级置信度 + 覆盖扩充，F 从 2.31 到 2.54。再往后每一轮 prompt 调整只带来 ~0.01 的边际收益。**纯 prompt 工程已到上限**。

16 个 F=1 的题目中，约 2/3 的裁判备注是「无资料支持」——检索没有找到正确文档。这是检索质量问题，不是 prompt 问题。

### 2. 检索-生成断层
即使文档已覆盖某 topic，检索也不一定能找到正确 chunk。原因是：
- BM25 关键词匹配对口语化问题（"挂了怎么办"→"补考"）效果差
- BGE-M3 向量检索对长文档的语义定位不够精确
- Reranker 虽然改善了排序，但不能创造正确 chunk（只能从候选中挑）

### 3. 评测集盲区
144 题中有 28 道是今天新加的，gold_source_ids 为空——裁判无法判断 faithfulness。实际有 gold 标注的约 116 题。新题需要人工标注 gold sources。

## 四、待讨论的问题

### Q1: 如何突破 Faithfulness 天花板？
当前 F=2.54，目标 3.5+。纯 prompt 已到极限。在不换模型的前提下，哪些方向还有 0.5+ 的提升空间？
- 检索增强（query expansion / HyDE / 多轮检索）？
- LLM 后验证（生成后用 Qwen 自己复查事实性）？
- 还是接受 2.5 是 8B 模型的天花板？

### Q2: 训练数据如何高效积累？
ROADMAP 要求 5K 对才能 LoRA fine-tune，当前只有 532 对（+ eval 标注）。最快路径：
- **LLM 自动生成**：让 Qwen 为 146 份文档各生成 3-5 个 QA 对 → ~600 对，加上现有 ≈ 1100 对
- **检索日志挖掘**：从线上 QA 日志中提取用户问题，人工标注检索结果
- **人工标注**：找同学标 500-1000 对

哪种优先级最高？5000 对的门槛是否合理（128 条就 fine-tune 成功的论文也存在）？

### Q3: 评测体系如何完善？
- 144 题中 28 道没有 gold_source_ids，需要标注
- 现有 gold 标注每题仅 1-2 个 source，recall@10 天花板低
- 是否需要引入人工评估（而非纯 LLM-as-judge）建立可信的 ground truth？

### Q4: 下一步优先级排序
在当前约束下（不换模型、16GB VRAM、WSL2），以下方向的 ROI 排序应该是什么？
1. LLM 自动生成训练数据 + LoRA fine-tune（等攒够数据）
2. 检索增强（query rewrite 改进 / HyDE / multi-hop）
3. LLM 后验证（生成后自查事实性）
4. 继续人工补文档 + 标注 gold sources
5. 接受当前质量，聚焦部署和用户体验

---

## 五、环境

| 项目 | 规格 |
|------|------|
| GPU | RTX 4070 Ti Super 16GB |
| LLM | Qwen3-8B (no-think, Ollama) |
| Embedding | BGE-M3 (sentence-transformers) |
| Reranker | BGE-Reranker-v2-m3 (Cross-Encoder) |
| OS | WSL2 Ubuntu on Windows 11 |
| Python | 3.12, sentence-transformers 5.5.1 |
