# 延迟优化报告：vLLM vs 退回 Windows

> 2026-06-06 · 当前状态：WSL2 Ollama 0.30.5，144 题 avg 4.80s，目标 ≤3.5s

## 1. 当前配置与问题

### 硬件

| 项目 | 值 |
|------|-----|
| GPU | RTX 4070 Ti Super 16GB, CC 8.9 |
| CUDA | 12.4, Driver 551.52 |
| CPU | WSL2 宿主机 (Windows 11) |

### 延迟分解（单次请求，正常路径 3.5-4.0s 的题）

| 阶段 | 耗时 | 占比 | 备注 |
|------|------|------|------|
| retrieve (Hybrid: BM25+Vector) | ~120ms | 3% | 稳定，不优化 |
| rerank (BGE-Reranker, top-20) | ~400ms | 10% | 已从 40→20，省 ~350ms |
| generate: **prefill** | ~1500-2000ms | 40% | **固定开销，与输出长度无关** |
| generate: **decode** | ~1700-2200ms | 45% | 89 t/s × 150-200 tokens |
| 其他 (classify, format 等) | ~50ms | 2% | 忽略 |
| **合计（正常路径）** | **~3.8-4.7s** | | |
| **合计（含 50% 重试率）** | **~4.80s** | | 空返回重试翻倍延迟 |

### 核心发现

延迟瓶颈不是输出长度（路线图 B 的判断错了），而是 **generate 阶段的固定 prefill 开销**。Qwen3-8B 在 WSL2 下对 ~1500 字符上下文的 prefill 需要 1.5-2s，这与输出 token 数无关。即使输出 0 tokens，prefill 照样要跑。

**6 月 2 日的 3.11s 基线是 Windows 原生 Ollama 跑的**，不是 WSL2。迁移到 WSL2 后 prefill 变慢是 GPU 直通（GPU-PV）的固有损耗。

```
▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊  prefill  (1.5-2.0s) ← 这个在 WSL2 下砍不掉
▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊▊  decode   (1.7-2.2s)
▊▊▊▊▊  rerank  (0.4s)
▊  retrieve (0.12s)
────────────────────────────
最少 3.8s，平均 4.8s
```

### 当前额外的坏配置

1. **num_predict=250 太紧** — ~50% 请求触发空返回，需要 retry（翻倍延迟）。根源是 /no_think 模板里空的 `<think></think>` 块也在消耗 token 预算
2. **模板用 /no_think 但不干净** — 模板末尾注入 `<think>\n\n</think>\n\n`，Qwen3 仍然会为这个空 think 块消耗 decoding budget
3. **BGE 模型和 LLM 抢显存** — BGE-M3 2.2GB + BGE-Reranker 1GB = 3.2GB，Ollama Qwen3-8B 5.2GB，总计 ~8.5GB。但加上 KV cache 和碎片，free VRAM 常不到 1.5GB，触发 empty_cache

## 2. 方案 A：vLLM

### 原理

vLLM 替代 Ollama 作为推理引擎。关键优势：

- **Prefix Caching**: 所有请求共享 system prompt 的 KV cache，prefill 只跑一次
- **Continuous Batching**: 并发请求可以共享 decode step
- **PagedAttention**: 显存利用率更高，碎片更少
- **FP16/BF16 原生推理**: 比 Ollama 的 Q4_K_M 量化精度更高

### 预期延迟

| 阶段 | Ollama (当前) | vLLM (预估) | 节省 |
|------|-------------|------------|------|
| prefill (首次) | 1500-2000ms | ~800ms (BF16, CUDA native) | ~1s |
| prefill (缓存命中) | 1500-2000ms | ~10ms (prefix cache) | ~1.9s |
| decode (per token) | 11.2ms (89 t/s) | ~8ms (125 t/s) | ~30% |
| decode (200 tokens) | 2240ms | 1600ms | ~640ms |
| **单请求延迟（无缓存）** | ~4.5s | **~2.8s** | -1.7s |
| **单请求延迟（缓存命中）** | ~4.5s | **~2.0s** | -2.5s |
| **144 题平均（含重试）** | 4.80s | **~2.5-3.0s** | -1.8~2.3s |

> Prefill 估算：vLLM 在 RTX 4070 Ti Super 上对 Qwen3-8B 的 prefill throughput 约 2000 tok/s（BF16），1500 token prompt ≈ 750ms。
> Decode 估算：vLLM BF16 decode 约 125 tok/s（vs Ollama Q4_K_M 的 89 tok/s）。

### 显存预算

| 组件 | 大小 |
|------|------|
| Qwen3-8B (BF16 weights) | ~15.2 GB |
| KV Cache (8192 ctx, BF16) | ~2.0 GB |
| vLLM overhead | ~0.5 GB |
| BGE-M3 embedding | ~2.2 GB |
| BGE-Reranker | ~1.0 GB |
| **合计** | **~20.9 GB** |

**严重超限！** 16GB 装不下 BF16 的 Qwen3-8B + BGE 模型。

### 解决方案

**2a. FP8 量化（vLLM 原生支持）**

```
pip install vllm
vllm serve Qwen/Qwen3-8B --quantization fp8 --max-model-len 8192 --gpu-memory-utilization 0.85
```

| 组件 | 大小 |
|------|------|
| Qwen3-8B (FP8) | ~8.0 GB |
| KV Cache | ~1.0 GB |
| vLLM overhead | ~0.5 GB |
| BGE-M3 | ~2.2 GB |
| BGE-Reranker | ~1.0 GB |
| **合计** | **~12.7 GB** |

16GB 可行，剩余 ~3.3GB 余量。

**2b. 将 BGE 模型移到 CPU（配合方案 2a）**

当前 `RERANKER_DEVICE=cpu` 已验证可行，rerank 延迟 200-500ms（vs GPU 的 400ms），差异不大。

| 组件 | 大小 |
|------|------|
| Qwen3-8B (FP8) + vLLM | ~9.5 GB |
| BGE-M3 (CPU) | 0 |
| BGE-Reranker (CPU) | 0 |
| **合计** | **~9.5 GB** |

剩余 ~6.5GB，非常安全。代价是 embedding 和 rerank 走 CPU。

### 部署步骤

```bash
# 1. 安装
source .venv/bin/activate
pip install vllm

# 2. 检查 FP8 支持（Ada Lovelace, CC 8.9 — 原生支持）
python -c "import torch; print(torch.cuda.is_bf16_supported())"

# 3. 下载模型（或使用本地 merged 版本）
# vLLM 可以直接加载 data/models/Qwen3-8B-NJU-LoRA-v3/

# 4. 启动（FP8 + LoRA）
vllm serve /home/mrtree/nju-rule-rag/nju-rule-rag/data/models/Qwen3-8B-NJU-LoRA-v3 \
    --quantization fp8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8001 \
    --dtype bfloat16

# 5. 修改 .env 指向 vLLM
# LLM_BASE_URL=http://localhost:8001/v1
# LLM_MODEL=Qwen3-8B-NJU-LoRA-v3
```

### 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| FP8 精度损失导致 F 下降 | 中 | 高 | 先 A/B 测试 30 题对比 Ollama |
| vLLM 安装/兼容问题 | 中 | 中 | venv 隔离，随时可回退 |
| LoRA adapter 加载失败 | 低 | 中 | 有 merged 模型，不需要 LoRA |
| 显存 OOM（FP8 不够） | 低 | 中 | BGE 切 CPU |
| 并发场景下稳定 | — | — | 当前无并发需求，1 req/s |

### 不可行的情况

如果 FP8 推理速度不达预期（< 100 tok/s decode），vLLM 的优势就不明显。需要实测。

---

## 3. 方案 B：退回 Windows 原生 Ollama

### 原理

6 月 2 日的 3.11s 是在 Windows 原生 Ollama 0.30.4 上跑出来的。Windows 上 CUDA driver 直接管理 GPU，没有 WSL2 GPU-PV 的中间层损耗。

### 预期延迟

| 指标 | WSL2 当前 | Windows 预期 |
|------|----------|-------------|
| prefill | 1500-2000ms | ~1000ms |
| decode | 89 t/s | ~100 t/s (CUDA 原生) |
| 单请求延迟 | 4.80s avg | **~3.0-3.5s** |
| 空返回重试率 | ~50% | 待验证（Windows 上 num_predict=250 是否稳定） |

### 部署步骤

```powershell
# Windows 端（PowerShell 管理员）
cd C:\Users\Mr.tree\AppData\Local\Programs\Ollama
.\ollama.exe serve

# 设置环境变量
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_KV_CACHE_TYPE=q8_0
set OLLAMA_KEEP_ALIVE=24h

# 拉取/创建模型（从 WSL2 复制 modelfile）
ollama pull qwen3:8b
ollama create qwen3:8b-nothink -f modelfile.qwen3-nothink
```

WSL2 端只需改 .env：

```bash
LLM_BASE_URL=http://<Windows_IP>:11434/v1  # WSL2 中宿主 IP 通常为 172.x.x.x
LLM_MODEL=qwen3:8b-nothink
```

### 优势

- **零安装成本** — Windows Ollama 已存在
- **已验证** — 6/2 的 3.11s 是真实数据
- **简单** — 只改一行 .env
- **风险低** — 随时可切回 WSL2

### 风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 上次 Windows 测试 11.8s | 高 | — | 根因是 num_predict=400 + 冷启动，已修复 |
| WSL2→Windows 网络延迟 | 低 | 低 | localhost proxy 或直接 localhost |
| Windows Ollama 稳定性 | 中 | 中 | 0.30.4 已验证稳定 |
| 模型 blob 需重新下载 | 低 | 低 | 已有 Windows .ollama 目录，SHA256 相同 |
| Dev 环境割裂（代码在 WSL2，LLM 在 Windows） | 低 | 低 | 仅 LLM 调用跨 OS，检索/embedding 仍在 WSL2 |

### Windows Ollama 上次 11.8s 的根因

诊断报告里的 Windows 测试只有一个数据点（P50=11.8s），且是在 num_predict=400 的旧配置下跑的。原因：

1. Windows Ollama 冷启动（模型未缓存到内存）
2. num_predict=400 → 输出 ~441 tokens（话痨模式）
3. 可能存在 Windows 端的环境变量未设置（Flash Attention 等）

**修复后重新测试**：num_predict=250（modelfile）+ OLLAMA_FLASH_ATTENTION=1 + 预热，预期回到 3.0-3.5s 范围。

---

## 4. 对比矩阵

| 维度 | WSL2 Ollama (当前) | vLLM FP8 | vLLM FP8 + BGE CPU | Windows Ollama |
|------|-------------------|----------|---------------------|----------------|
| 延迟（144Q avg） | 4.80s | **~2.8s** | **~3.0s** | **~3.2s** (待验证) |
| 安装成本 | 已完成 | 中（pip + 配置） | 中 | **零** |
| 显存安全 | 紧张 | 紧张（12.7/16） | **宽裕（9.5/16）** | 同当前 |
| 模型精度 | Q4_K_M (4-bit) | FP8 | FP8 | Q4_K_M (4-bit) |
| LoRA 支持 | GGUF 合并 | 原生 | 原生 | GGUF 合并 |
| Prefix 缓存 | 无 | **有** | **有** | 无 |
| 回退复杂度 | — | 改一行 .env | 改一行 .env | 改一行 .env |
| 风险 | — | 中（FP8 精度、兼容性） | 低 | **最低** |

---

## 5. 推荐路线

### 首选：Windows Ollama（方案 B）

**理由**：
1. 零安装成本，Windows Ollama 和模型权重都已经就绪
2. 6 月 2 日 3.11s 的基线已验证可行
3. 当前延迟问题 80% 来自 WSL2 GPU-PV 的 prefill 损耗，回 Windows 是直击根因
4. 风险最低，随时可切回

**行动**：
1. 在 Windows 端启动 Ollama，设置 Flash Attention + Q8 KV Cache
2. 用当前 modelfile 创建 qwen3:8b-nothink（num_predict=250）
3. WSL2 .env 改 `LLM_BASE_URL` 指向 Windows IP
4. 跑 144 题 eval，看延迟和关键词命中
5. 如果 < 3.5s 且 F 不跌 → 收工

### 次选：vLLM FP8 + BGE CPU（方案 A-2b）

**理由**：
1. 如果 Windows 不及预期（例如 < 3.5s 但不稳定），vLLM 是真正的长期方案
2. FP8 精度比 Q4_K_M 高，F 分可能提升
3. Prefix caching 让重复请求接近零 prefill
4. BGE 切 CPU 解决显存问题，且 rerank 延迟影响不大（200-500ms）

**行动**：
1. `pip install vllm`
2. 用本地 merged 模型 + FP8 量化启动
3. A/B 测试 30 题对比 Ollama
4. 确认延迟和 F 都达标后全量切换

### 不做：vLLM BF16（方案 A-1）

16GB 显存放不下 Qwen3-8B BF16 + BGE 模型。不考虑。

---

## 6. 当前代码已做的优化（保留）

无论选哪个方案，以下改动都保留：

| 改动 | 文件 | 效果 |
|------|------|------|
| num_predict 400→250 | modelfiles | 限制输出长度 |
| RERANK_CANDIDATE_K 40→20 | .env | rerank 省 ~350ms |
| 空返回自动 retry | pipeline.py | 容错 |
| 系统提示词优化 | config.py | 已回退到原始（稳定版） |

如果切到 Windows 或 vLLM 后空返回率降到 ~0%，可以考虑去掉 retry 以省掉重试开销。
