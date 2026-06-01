# NJU Rule RAG — 路线选择报告

> 2026-06-02 | 完整评测对比 + 两条路线分析

## 当前状态总览

| 模型/路线 | Faithfulness | Relevance | Refusal | Overall | 延迟 |
|-----------|-------------|-----------|---------|---------|------|
| v0.5.2 基线 (qwen3:8b-nothink) | 2.31 | 3.88 | 4.48 | 3.56 | 2.23s |
| v2 LoRA (nju-lora-v2) | **2.01** | **1.96** | 4.24 | 2.74 | 4.92s |
| **v3 路线B (fact_check+prompt+分流)** | **2.94** | **3.28** | **4.86** | **3.69** | 3.11s |

> Judge: DeepSeek-chat（三个评测统一用同一法官，可对比）

## 已完成的工作

### 路线 B（系统层优化，已部署）

三件事：

**B.1 输出层事实核查** (`app/fact_check.py` + pipeline 集成)
- 正则提取答案中的数字/日期/URL/电话/金额等事实
- 与检索到的原文逐字比对（布尔约束，100% 准确）
- 未命中处理：单条→对冲句；多条→降级 Tier 3 模板回复
- **效果：F 从 2.31→2.94 (+27%)**

**B.2 Prompt v4 简化** (`app/config.py`)
- 从 1267 字符精简到 558 字符 (56% shorter)
- 删除「不编造数字/流程/网址」等重复规则（B.1 程序级护栏替代）
- 保留核心身份 + 好/坏示例
- **效果：Relevance 从 1.96→3.28（恢复到接近基线）**

**B.3 高风险题分流到 DeepSeek** (`app/pipeline.py`)
- risk_level=high → 自动走 DeepSeek fallback
- 低中风险题保持本地 qwen3:8b-nothink
- **效果：高风险题不再被本地模型低质量处理**

### 修复的 Bug

| Bug | 影响 | 修复 |
|-----|------|------|
| PyTorch 2.12+cu130 vs 驱动 CUDA 12.4 | GPU 不可用 | 重装 torch 2.6+cu124 |
| nju-lora-v2 GGUF 缺 chat template | 重复输出 | Ollama Modelfile 嵌入 Qwen3 template |
| nju-lora-v2 Q8_0 8.7GB 撑爆显存 | eval 到 90 题后 OOM | 重做 Q4_K_M 量化 (4.7GB) |
| eval_generation.py context 空 | 法官评分为假象 | 从 chunks.jsonl 按 chunk_id 加载内容 |
| .env FALLBACK_LLM_MODEL 指向 nju-lora-v2 | 回退失效 | 改回 deepseek-chat |

## v2 LoRA 失败分析

### 为什么 v2 不如基线

v2 label masking 修复正确，loss 曲线健康，但 **F=2.01 低于基线的 2.31**。原因：

1. **没有通用回放数据**：3853 条 NJU 样本训练 1 epoch，模型过拟合到校规域，丢失了部分通用对话能力。ROADMAP_LORA_V3 要求 15% 回放但未实现
2. **没有验证集 + 早停**：`eval_strategy="no"`，305 条 holdout 从未使用。训练在 241 步就停了（1 epoch），但不知道何时是最优检查点
3. **训练时评估已关闭**：无法判断 loss 拐点

### 关于基座模型的纠正

ROADMAP_LORA_V3 诊断「v2 失败根因是用了 Qwen3-8B Base 而非 Instruct」——**这个判断有误**。HuggingFace 上不存在 `Qwen/Qwen3-8B-Instruct` 仓库。Qwen3-8B 自带 chat template (4168 字符)，内建指令遵循能力。v2 失败是训练工艺问题（缺回放+验证），不是基座选择问题。

## 路线 A 剩余可行方案

既然 Qwen3-8B-Instruct 不存在，路线 A 改为：

**在现有 Qwen3-8B 基座上重训 LoRA，补全缺失项：**
1. 加 600-800 条通用回放数据 (alpaca-zh)，占训练集 15%
2. 用 305 条 lora_holdout.jsonl 做验证集，每 50 步 eval
3. 加 EarlyStoppingCallback(patience=3)
4. 训前 dry-run 做 token 级 label 检查
5. 训后冒烟测试通过再进 GGUF
6. 基座、rank、lr 不变（已验证合理）

**预期**：F 应在 2.5-3.0 之间（通用回放防遗忘 + 早停防过拟合）。但不会超过路线 B 的 2.94，因为路线 B 有程序级护栏。

**预算**：1-1.5 天

## 建议

**两条路不互斥，可以都做**：
1. 当前路线 B (F=2.94) 已是可用状态，可以上线
2. 路线 A 补上通用回放 + 早停后重训，目标是 ≥ 路线 B 的 2.94
3. 如果路线 A 成功，用 LoRA 模型替换本地 qwen3:8b-nothink，B.1 事实核查继续作为后处理护栏
4. 长远：等数据飞轮攒到 1.5 万+ 真实 Q&A 对再做大模型微调

## 关键文件

| 文件 | 内容 |
|------|------|
| `app/fact_check.py` | 事实核查模块 |
| `app/config.py:16-45` | 简化后的 system prompt |
| `app/pipeline.py:147-150` | 高险分流逻辑 |
| `app/pipeline.py:162-187` | fact_check 集成点 |
| `scripts/eval_generation.py:62-96` | 修复后的 context 加载 |
| `scripts/modelfile.nju-lora-v2` | Ollama 模板（含 chat template） |
| `scripts/apply_v3_changes.py` | 路线 B 变更脚本 |
| `data/eval/ROADMAP_A_plan.md` | 路线 A 详细计划 |
