# LoRA v2 最终评测报告

> 2026-06-01 | 从 C 盘迁移到 D 盘后，完成 v2 模型全部评测

## 一、迁移修复清单

| # | 问题 | 修复 |
|---|------|------|
| 1 | PyTorch 2.12.0+cu130 与 CUDA 12.4 驱动不匹配 | 重装 torch==2.6.0+cu124 |
| 2 | `.env` 中 FALLBACK_LLM_MODEL 误设为 `nju-lora-v2` | 改回 `deepseek-chat` |
| 3 | Ollama nju-lora-v2 GGUF 缺少 chat template（`TEMPLATE {{ .Prompt }}`）导致推理格式与训练格式不匹配，模型输出大量重复 | 用 Go template 语法写入 Qwen3 chat template |
| 4 | nju-lora-v2 Q8_0 (8.7GB) + BGE-M3 (2.2GB) + BGE-Reranker (1.0GB) + KV cache 撑爆 16GB 显存，eval 到 90 题后连续 TIMEOUT → GPU OOM | 用 llama.cpp 从 FP16 合并模型重做 Q4_K_M 量化（8.7GB→4.7GB），RERANKER_DEVICE 保持 auto |

## 二、v2 训练执行情况 vs ROADMAP_LORA_V2

| 项目 | 计划 | 实际 | 状态 |
|------|------|------|------|
| Label masking (`labels[prompt]=-100`) | 必须 | 已正确实现 | ✅ |
| Dry-run (50样本验证) | 必须 | 跳过 | ⚠️ |
| 通用数据回放 15% | 必须 | 未实现 | ❌ |
| eval 验证集监控 | 每50步 | `eval_strategy="no"` | ❌ |
| Token级 label 打印 | 第0步 | 仅计数检查 | ⚠️ |
| 冒烟测试 (5 canary) | GGUF前 | 跳过（GGUF后才补做） | ⚠️ |
| 全量评测 | 必须 | 已完成 | ✅ |

### 训练日志 (trainer_state.json)

```
Step  10: loss=2.58  grad_norm=0.059
Step  50: loss=1.78  grad_norm=0.029
Step 100: loss=1.60  grad_norm=0.034
Step 150: loss=1.55  grad_norm=0.040
Step 200: loss=1.57  grad_norm=0.045
Step 241: loss=1.54  grad_norm=0.043  (1 epoch 完成)
```

- Loss 从 2.58→1.54，健康（v1 是 10.7→0.17）
- Grad norm 0.03-0.05，健康（v1 是 0.002）
- 但终点 loss 1.54 高于 roadmap 预期的 0.5-1.0 范围

## 三、评测对比

| 指标 | v0.5.2 基线 (qwen3:8b-nothink) | v1 (label masking bug) | v2 (修复后) |
|------|------|------|------|
| 端到端成功率 | 100% | 未评测 | **100% (144/144)** |
| 平均延迟 | 2.23s | — | 5.13s |
| 来源覆盖率 | 100% | — | **100%** |
| 关键词命中 | 95.8% | — | 68.8% |
| recall@5 | 0.881 | — | 0.824 |
| MRR | 0.612 | — | 0.619 |
| Context Precision@10 | 0.12 | — | 0.128 |
| **Faithfulness** | **2.31** | 0.0 | **1.74** |
| **Relevance** | **3.88** | — | **1.98** |
| **Refusal Correctness** | **4.48** | — | **4.69** |
| **Overall** | **~3.56** | 0.0 | **2.80** |

注：v2 延迟翻倍是因为 Q4_K_M 生成速度（74 tok/s）叠加回答更长（~300 tokens/题），原始管道耗时 2×。

## 四、已知缺陷（v2 训练脚本 `lora_train_v2.py`）

1. **基座模型错误**：用的是 `Qwen3-8B`（预训练基座），不是 `Qwen3-8B-Instruct`。基座模型没有指令遵循能力，4583 条样本不足以从零教会它
2. **无通用回放**：脚本完全没有加载通用对话数据，roadmap 要求的 15% 回放未实现
3. **无验证集**：`eval_strategy="no"`，305 条 holdout 从未被使用，无法监控过拟合
4. **无早停**：依赖手动判断
5. **CLI 参数 bug**：`--epochs=` 和 `--max_samples=` 参数解析有误（`max_n = max_n` 是 no-op）
6. **合并模型时 chat template 未嵌入 tokenizer_config.json**，导致 GGUF 丢失模板（已手动修复）

## 五、关键发现

**v2 比 v1 有本质改善**（v1 输出全是训练模板字符，F=0），但 **Faithfulness 和 Relevance 反而比不用 LoRA 的原始 qwen3:8b-nothink 更差**。

唯一改善的指标是 Refusal Correctness（4.48→4.69），说明 label masking 修复让模型学会了「不知道就说不知道」。

检索指标（recall@5, MRR, Context Precision）基本持平——检索器没变。

## 六、文件清单

| 文件 | 说明 |
|------|------|
| `data/lora_adapters/nju-v2/` | LoRA adapter (61MB) + checkpoint-241 |
| `data/models/qwen3-8b-nju-lora-v2-Q4_K_M.gguf` | Q4_K_M GGUF (4.7GB) — **当前使用** |
| `data/models/qwen3-8b-nju-lora-v2.gguf` | Q8_0 GGUF (8.7GB) — 太重已废弃 |
| `data/models/qwen3-8b-nju-lora-v2-fp16.gguf` | FP16 GGUF (16.4GB) — 中间产物可删 |
| `data/models/Qwen3-8B-NJU-LoRA-v2/` | 合并后的 HF 模型 (fp32, 16.3GB) |
| `data/models/Qwen/Qwen3-8B/` | 基座模型（注意：是 base 不是 Instruct） |
| `scripts/modelfile.nju-lora-v2` | Ollama Modelfile（已含 chat template） |
| `scripts/lora_train_v2.py` | v2 训练脚本 |
| `data/eval/results.csv` | eval_rag 结果 (144 题) |
| `data/eval/gen_scores.csv` | eval_generation 评分 (144 题, DeepSeek judge) |
