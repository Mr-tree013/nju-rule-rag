# LoRA v1 训练失败报告

## 做了什么

用 3853 对 NJU 校规 QA 数据，QLoRA（rank=16, lr=1e-4）微调 Qwen3-8B，2 epochs，4.5 小时，VRAM 15.5GB/16GB。

数据构成：50% 充分回答 + 25% 部分回答（含对冲） + 20% 拒答 + 5% 通用对话。

## 结果：F=0.0

模型输出严重重复，典型表现：

```
问：重修要不要额外交钱？
答：怎么交？

【参考答案】
重修课程学分学费的缴纳，需在重修申请学期开
学初规定的时间内，按课程学分收费标准缴纳课程学分学费...
【参考答案】
重修课程学分学费的缴纳，需在重修申请学期开
学初规定的时间内，按课程学分收费标准缴纳课程学分学费...
【参考答案】...（无限重复）
```

## 已确认不是问题的地方

- **pipeline 没问题**：数据生成→训练→合并→GGUF→Ollama 全链路跑通
- **显存没问题**：15.5GB/16GB，没有 OOM
- **训练稳定性没问题**：grad_norm 始终在 0.002，loss 平滑下降
- **DeepSeek 生成答案没问题**：validation 时人工抽检通过

## 需要诊断的问题

### Q1: 为什么会重复输出模板？

训练数据全部使用 Qwen3 chat template 格式：
```
<|im_start|>system
你是南大学长...
<|im_end|>
<|im_start|>user
问题：XXX
<|im_start|>assistant
【参考答案】...
```

推理时也用了相同的 system prompt。模型在 3853 条同格式数据上训了 2 epoch，loss 从 10.7 降到 0.17——严重过拟合。它学会了「assistant 之后就应该输出模板格式」，而不是「根据问题自然回答」。

**核心疑问**：这是训练格式的问题（chat template 不适合指令微调），还是 epoch 太多的问题，还是数据多样性不够？

### Q2: 2 epochs 是否太多？

Loss 曲线：start=10.7 → end=0.17。ROADMAP_LORA 说"1-1.5 epoch 达最优"。我们用的是 2 epochs，但 loss 在 epoch 1 时已经降到 ~0.5。可能 1 epoch 就够，甚至 0.5 epoch。

但 ROADMAP_V0.6 §3.3 又说"2 epoch，不是 3"。这个建议是针对不同数据量的吗？

### Q3: 训练数据格式是否正确？

当前格式：instruction-tuning（system + user + assistant）。但 Qwen3 的官方推荐微调格式是什么？是否需要添加 `add_generation_prompt`？是否需要做 label masking（只对 assistant 部分计算 loss）？

我们的训练脚本使用 `tokenizer.apply_chat_template()` 自动格式化，然后用 `labels = input_ids.copy()`——这意味着模型对所有 token（包括 system prompt 和 user prompt）都计算 loss。这可能是模型学会"复制"模板的原因。

### Q4: LoRA rank 和 target_modules 是否正确？

当前 rank=16, target=q_proj/k_proj/v_proj/o_proj。ROADMAP_LORA 建议这个配置。但 0.19% 的参数更新比例对于 3853 条数据来说可能太多了——rank=8 或 target 只选 q_proj+v_proj 是否更好？

### Q5: 是否需要混入通用数据？

ROADMAP_LORA §1.2 说必须加 5% 通用数据（alpaca-zh）防遗忘。我们加了但可能不够——5% 只约 200 条。是否需要 10-20%？

而且这些通用数据的格式应该是什么？如果它们的格式也是 instruction-tuning，那格式多样性还是不够——所有的训练样本用的都是同一种 prompt 模板。

### Q6: 下一步应该怎么调？

在 ROADMAP_LORA 的框架下，最优先调整哪个参数？

- A: epoch 减到 1，其他不变
- B: epoch=1 + rank 降到 8
- C: epoch=1 + 混入 20% 通用数据
- D: 改造训练格式（不用 chat template，用更简单的 prompt 格式）
- E: 先不做全量训练，用 500 条数据做网格搜索

## 环境

- GPU: RTX 4070 Ti Super 16GB
- Base model: Qwen3-8B（ModelScope 下载的 PyTorch 权重）
- QLoRA: 4-bit NF4, peft + bitsandbytes
- 训练框架: transformers Trainer
- 部署: merge → GGUF (Q8_0) → Ollama
