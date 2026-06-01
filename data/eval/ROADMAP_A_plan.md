# ROADMAP_A — 换 Instruct 基座重训 LoRA 执行计划

## 前置检查

### ✅ 已有
- lora_train.jsonl: 3853 条 NJU Q&A（含 answer 字段）
- lora_holdout.jsonl: 305 条（含 answer，可直接当 eval set）
- Label masking 代码已验证正确（v2 训练脚本）
- Ollama Modelfile 模板（scripts/modelfile.nju-lora-v2）
- Chat template 修复方案已验证
- 合并 + GGUF + deploy 流程已知

### ❌ 缺失（需准备）
1. **Qwen3-8B-Instruct 基座模型**: 不在 HF 缓存中，需从 HuggingFace 下载（~16GB）
2. **通用回放数据**: 无 alpaca-zh/belle 类数据，需下载 ~600-800 条并格式对齐
3. **训练脚本升级**: 当前 lora_train_v2.py 缺少 eval set/early stopping/general replay/token sanity check

### ⚠️ 需验证
- Qwen3-8B-Instruct 是否与 Ollama 的 `qwen3:8b-nothink` 同源（先做基线评测对比）
- HuggingFace 是否有 `Qwen/Qwen3-8B-Instruct`（确认存在再下载）

## 执行步骤

### Step 0: 基座一致性验证（30 分钟）
```
1. 下载 Qwen3-8B-Instruct（如 HF 有）
2. 不做 LoRA，直接用 Instruct 做 GGUF + Ollama 部署
3. 跑 144 题 eval_rag + eval_generation
4. 对比当前 qwen3:8b-nothink 基线（F=2.31）
   - 差 ≤ 0.2 → 同源，可继续
   - 差 > 0.2 → 基座不同源，需要解决一致性问题
```

### Step 1: 准备通用回放数据（30 分钟）
```
1. 下载 alpaca-zh 或 belle_10k（HuggingFace）
2. 采样 600-800 条多样风格短指令
3. 用 Qwen3 chat template 重新格式化
4. 存为 data/training/general_replay.jsonl
```

### Step 2: 写 lora_train_v3.py（1 小时）
改动点（基于 v2 脚本）:
```
1. MODEL_ID = "Qwen/Qwen3-8B-Instruct"      # ← 关键修改
2. 加载 general_replay.jsonl，按 15% 比例混入训练集
3. 加载 lora_holdout.jsonl 作为 eval_dataset
4. TrainingArguments 加 eval_strategy="steps", eval_steps=50,
   load_best_model_at_end=True, metric_for_best_model="eval_loss"
5. Trainer 加 EarlyStoppingCallback(patience=3)
6. 加 LossSanityCallback（loss < 0.3 或 > 5.0 报警）
7. 第 0 步打印 token-level labels（肉眼确认 mask）
8. 修复 CLI 参数: --debug, --max_samples=N, --epochs=N
9. 输出目录: data/lora_adapters/nju-v3/
```

### Step 3: Dry-run 验证（30 分钟）
```
python scripts/lora_train_v3.py --debug --max_samples=50 --epochs=1

必须通过的检查点:
- ✅ 第 0 步 token labels: system+user 全 MASKED, assistant 全 trained
- ✅ 第 0 步 loss 在 2.0-4.0（Instruct 基座应已会对话，loss 比 Base 低）
- ✅ 第 50 步 loss 降到 1.0-1.5
- ✅ 显存稳定 ≤ 14.5GB
- 任一不过 → 停下来排查
```

### Step 4: 全量训练（2-3 小时）
```
python scripts/lora_train_v3.py --epochs=1 --general_replay_ratio=0.15

监控:
- 训练 loss step 100 前从 ~2.5 降到 ~1.2
- 验证 loss 与训练 loss 同步降，差距 ≤ 0.3
- 验证 loss 在 600-1000 步见底，Early stopping 触发
- 最终 holdout loss 在 0.5-0.9
```

### Step 5: 冒烟测试（5 分钟）
```
用 PeftModel 加载 adapter，跑 5 道 canary:
1. 重修要不要额外交钱？
2. 百团大战是什么？
3. 本校学生考研有什么优势？
4. 校园网怎么收费？
5. 今天天气怎么样？

检查: 无训练模板字符、不重复、不复述问题、不空输出/乱码
任一不过 → 不进 GGUF
```

### Step 6: GGUF + Ollama 部署（30 分钟）
```
1. merge_lora.py → 合并 adapter 到 base model
2. convert_hf_to_gguf.py → FP16 GGUF
3. llama-quantize → Q4_K_M (~5GB)
4. ollama create + Modelfile（带 chat template）
5. 快速验证 3 题确认部署正确
```

### Step 7: 全量评测（30 分钟）
```
eval_rag.py (144 题) + eval_retrieval.py + eval_generation.py (--deepseek)
```

## 验收红线（V3 §3.1）

| 指标 | v3 红线 | 当前基线 |
|------|---------|---------|
| Instruct 不带 LoRA 的 F | ±0.2 内 vs 当前基线 | Step 0 验证 |
| 训练 loss 终点 | 0.5-1.0 | v2 是 1.54 |
| holdout/train loss | ≤ 1.5 | — |
| **Faithfulness** | **≥ 3.0** | 2.94 |
| **Relevance** | **≥ 3.6** | 3.28 |
| Refusal Correctness | ≥ 4.5 | 4.86 |
| 通用对话 20 题 | 不比 base 差 | — |
| 关键词命中 | ≥ 90% | 76.4% |

## 预算

| 步骤 | 时间 |
|------|------|
| Step 0: 基座验证 | 0.5h |
| Step 1: 回放数据 | 0.5h |
| Step 2: 训练脚本 | 1h |
| Step 3: Dry-run | 0.5h |
| Step 4: 全量训练 | 2-3h |
| Step 5: 冒烟测试 | 5min |
| Step 6: GGUF 部署 | 0.5h |
| Step 7: 全量评测 | 0.5h |
| **总计** | **1.5-2 天** |

## 风险与止损

- 如果 Step 0 基座一致性验证失败 → 解决基座问题再训
- 如果 Dry-run token labels 检查失败 → 排查 label masking
- 如果训练 loss 终点 > 1.0 → epoch 不够或 lr 太小，调整后重训
- 如果 F < 2.94（低于当前路线 B） → 不上线，回退到路线 B
- 如果 Relevance < 3.6 → 结构性失败，排查训练数据
