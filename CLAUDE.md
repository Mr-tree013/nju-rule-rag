# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NJU Rule RAG — a retrieval-augmented generation bot for Nanjing University undergraduate academic rules. Students ask questions in natural language; the system retrieves relevant regulatory documents and generates answers with source citations and risk-level classification.

**Current status**: v0.6.3. 147 source documents → 3248 chunks. 145 eval questions. Avg latency 2.23s. Fact-check pipeline (NER entity verification) added. LoRA v3 fine-tuning + vLLM/Ollama GGUF deployment. Confidence tiering (3 tiers, Tier 3 skips LLM), topic routing (soft source boost), LLM-as-Reranker option. Faithfulness 2.99/5. Known issue: context precision still low (0.12), limiting further faithfulness gains.

## Commands

```bash
source .venv/bin/activate

# Start server — one-click (auto-clears proxy, sets GPU env, runs preflight)
./scripts/start_server.sh
./scripts/start_server.sh --reload  # dev mode with auto-reload

# Preflight check (diagnose startup issues without starting server)
python scripts/preflight_check.py

# Run tests
pytest                                    # all tests
pytest tests/test_pipeline.py -x          # pipeline tests, stop on first failure
pytest tests/test_retriever.py            # retriever unit tests
pytest tests/test_config.py               # config tests
pytest tests/test_main.py                 # API endpoint tests (needs server)

# ── Data pipeline ──

PYTHONPATH=. python scripts/parse_to_markdown.py      # raw HTML/PDF → processed/*.md
PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py
python scripts/validate_training_data.py               # validate LoRA training pairs

# ── Evaluation ──

python scripts/eval_rag.py                  # 118-question /ask eval (needs server)
PYTHONPATH=. python scripts/eval_retrieval.py          # retrieval metrics (direct)
PYTHONPATH=. python scripts/eval_retrieval.py --rerank # with reranker
PYTHONPATH=. python scripts/eval_retrieval.py --rewrite # with query rewrite
PYTHONPATH=. python scripts/eval_generation.py         # LLM-as-judge scoring
PYTHONPATH=. python scripts/tune_weights.py           # weight grid search
PYTHONPATH=. python scripts/check_regression.py       # CI regression gate
python scripts/annotate_gold_sources.py                # refresh gold-source labels

# ── Feedback & ops ──
python scripts/review_feedback.py              # review negative feedback entries
python scripts/review_feedback.py --stats      # feedback summary only
python scripts/review_staging.py               # interactive staged document review
python scripts/review_staging.py --list        # list staged files
python scripts/crawl_sources.py                # one-shot crawl
PYTHONPATH=. python scripts/crawl_scheduled.py # scheduled crawl run

# ── Analysis ──
PYTHONPATH=. python scripts/error_taxonomy.py        # classify F<=2 answers
PYTHONPATH=. python scripts/coverage_gap_analysis.py # per-topic ROI analysis

# ── Stream test ──
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
  -d '{"question": "补考没过怎么办"}'
```

## Architecture

```
POST /ask {"question": "..."}
        │
        ▼
[_handle_meta_question]  "你是谁"/"你能干什么" → 直接回复（不走检索）
        │
        ▼
[QueryRewriter]        口语化规范化（should_rewrite()守卫，默认跳过正式问题）
        │
        ▼
TwoLayerRiskClassifier  L1关键词(高召回) → L2 BGE-M3 centroid消歧
        │  (BGE-M3 encode 受 GPU RLock 保护)
        │
        ▼
[_classify_topic]      题目主题分类 → 匹配 topic_route_map → soft boost 1.2×
        │               (非硬过滤，仅提升相关source分数)
        │
        ▼
HybridRetriever        BM25(0.25) + BGE-M3 Vector(0.45) + Priority(0.30)
        │  (BGE-M3 encode 受 GPU RLock 保护 — 与 classifier 共享同一锁)
        │
        ▼
CrossEncoderReranker   BGE-Reranker-v2-m3 (40候选→12精排)
        │  score融合: 0.4×原始 + 0.6×sigmoid(logit)
        │  或 LLM-as-Reranker (RERANKER_TYPE=llm)
        │  (CrossEncoder predict 受 GPU Lock 保护)
        │
        ▼
_filter → _dedup       score阈值过滤 → max 3/source, 12 total
        │
        ▼
[_decide_confidence_tier]  3-tier: T1自信回答 | T2轻度hedge | T3跳过LLM直接转人工
        │  Tier 3 → 跳过 LLM + fact check，直接返回转介回复
        │
        ▼
_build_prompt          token预算裁剪 (budget=4096, max 6 chunks, 320/chunk)
        │               + Tier 2自动注入hedge指令
        │               + 高风险题自动追加短系统补丁
        │
        ▼
LLM (Qwen3-8B)         temp=0.15, num_ctx=8192, num_predict=400, stop sequences
        │  timeout=20s → 超时自动 fallback→DeepSeek
        │  高风险 + ENABLE_HIGH_RISK_DEEPSEEK → 直接路由到 DeepSeek
        │  (HTTP I/O — 不加锁，可并发)
        │
        ▼
[fact_check]           NER实体校验（数字/日期/URL/金额）→ 删除无出处句或hedge
        │               3级惩罚: 硬删除 | hedge | Tier降级
        │               3+失败或硬失败 → 降级到 Tier 3
        │               (ENABLE_FACT_CHECK守卫, 默认开启)
        │
        ▼
[_verify_citations]    答案句bigram与来源重叠度检查（ENABLE_CITATION_VERIFY）
        │
        ▼
_format_response       长度截断(600字) + 高风险模板追加联系方式(NOT LLM生成)
        │               + 全链路timing打点写入 debug.timing
        │
        ▼
_maybe_free_gpu_cache  N次请求后empty_cache，空闲<1.5GB时强制释放
        │
        ▼
{ question, answer, risk_level, need_human_confirm, sources[], debug, confidence_tier, tier_top1, tier_top3 }
```

### Key module responsibilities

| File | Role |
|------|------|
| `app/main.py` | FastAPI entry point — all endpoints, CORS, QQ bot webhook |
| `app/pipeline.py` | `RAGPipeline` — orchestrates the full ask flow; each step is a method you can override |
| `app/config.py` | Frozen `Settings` dataclass from `.env`; system prompt with few-shot anti-hallucination examples |
| `app/retriever.py` | `HybridRetriever` — BM25 (0.25) + BGE-M3 vector (0.45) + priority boost (0.30); `VectorRetriever` serializes GPU via `threading.RLock` |
| `app/reranker.py` | `CrossEncoderReranker` and `LLMReranker` — BGE-Reranker-v2-m3 with score fusion (`0.4×raw + 0.6×sigmoid(logit)`) or LLM-based pointwise ranking; GPU-serialized via `threading.Lock` |
| `app/policy.py` | `TwoLayerRiskClassifier` — L1 keywords (high recall) + L2 BGE-M3 centroid disambiguation; also `classify_topic()` for topic routing; shares GPU lock with retriever |
| `app/llm_client.py` | OpenAI-compatible client with streaming, 3-retry backoff, 20s timeout → DeepSeek fallback |
| `app/fact_check.py` | NER entity verification — extracts numbers/dates/URLs/amounts from output, checks against sources; 3-tier penalty (hard-remove sentence / hedge "资料里没写" / downgrade to Tier 3); takes `confidence_tier` param |
| `app/query_rewriter.py` | Colloquial→formal query normalization; `should_rewrite()` guard, default off |
| `app/cache.py` | LRU in-memory QA cache (200 entries, 1h TTL) |
| `app/qq_bot.py` | QQ Bot adapter (NapCat/OneBot v11 HTTP callback) |
| `app/health.py` | Deep health check — Ollama, GPU, models, index, cache |
| `app/errors.py` | Structured error types for consistent API error responses |
| `app/deps.py` | Wiring — shares GPU lock from retriever to classifier, passes reranker device setting |
| `app/crawl/` | Document crawling and fetching for source acquisition |

## Model inventory

| Model | Size | Where | Purpose | Thread-safe? |
|-------|------|-------|---------|-------------|
| Qwen3-8B (no-think) | 5.2 GB | Ollama `qwen3:8b-nothink` | LLM generation (base model, no LoRA) | N/A (separate process) |
| Qwen3-8B LoRA v3 | 5.0 GB | Ollama `nju-lora-v3` (Q4_K_M GGUF) | Fine-tuned LLM on NJU QA pairs | N/A (separate process) |
| Qwen3-8B LoRA v3 (4-bit) | 6.0 GB | `data/models/Qwen3-8B-NJU-LoRA-v3/` | transformers/bnb 4-bit for vLLM | N/A |
| BGE-M3 | 2.2 GB | sentence-transformers | Query/document embedding (1024-dim) | **No** — serialized via GPU RLock |
| BGE-Reranker-v2-m3 | 1.0 GB | sentence-transformers | Cross-encoder reranking | **No** — serialized via GPU Lock |
| BGE-Reranker-NJU | 1.0 GB | `data/models/bge-reranker-nju/` | Fine-tuned reranker on NJU relevance pairs | **No** |

Total GPU memory: ~8-10 GB with base model, ~8 GB with v3 Q4_K_M (fits 16GB with ~6-8 GB headroom for BGE models). Create the no-think variant via `ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink`.

## Data pipeline

1. `data/sources.csv` — 147 source documents (priority 1-5, department, scope).
2. Raw documents: `data/raw/` (HTML/PDF/DOC) → `scripts/parse_to_markdown.py` → `data/processed/*.md`. New documents can also be staged in `data/staging/` and reviewed via `review_staging.py` before adding.
3. `scripts/build_chunks.py` — `data/processed/*.md` → `data/chunks/chunks.jsonl` (3248 chunks). Splits by article headings (including `**第X条**` bold markdown). 0 too-long chunks enforced via `_split_by_fixed_size` fallback.
4. `scripts/build_index.py` — BM25 (jieba) + Chroma (BGE-M3, 1024-dim). GPU auto-detection. `batch_size=8` for 16GB VRAM.

To add a document: `.md` → `data/processed/`, add row to `data/sources.csv`, then `build_chunks.py && build_index.py && validate_*`.

Chunk format and API response format are defined in `docs/dev_contract.md` — the contract between the data pipeline and the online service.

## LoRA fine-tuning & training pipeline

The project includes a full LoRA fine-tuning pipeline for Qwen3-8B on NJU-domain QA pairs:

**Training data** (`data/training/`):
- `lora_train.jsonl` — 3853 filtered QA pairs for LoRA training
- `lora_holdout.jsonl` — 305 holdout pairs for evaluation
- `all_pairs_filtered.jsonl` — 4158 pairs after quality filtering
- `paraphrased_pairs_r*.jsonl` / `synthetic_pairs_r*.jsonl` — LLM-generated augmentations

**Scripts**:
```bash
# Generate training data
PYTHONPATH=. python scripts/generate_training_pairs.py [--limit N]
PYTHONPATH=. python scripts/generate_training_answers.py

# Validate training data quality
python scripts/validate_training_data.py

# Train LoRA adapter (v2: correct label masking, prompt masked, only assistant trained)
python scripts/lora_train_v2.py [--debug] [--max_samples=N]

# Evaluate LoRA model (3-layer: canary A/B, holdout loss, full eval)
python scripts/eval_lora.py --layer1   # A/B canary test
python scripts/eval_lora.py --layer2   # holdout loss
python scripts/eval_lora.py --layer3   # full eval + report

# Fine-tune reranker on NJU-domain relevance pairs
python scripts/finetune_reranker.py
python scripts/finetune_reranker.py --eval-only

# Deploy LoRA model
bash scripts/start_vllm.sh nju-lora    # vLLM on port 8001 (uses nju-v3 adapter)
# Or quantize for Ollama:
# ollama create nju-lora-v2 -f scripts/modelfile.nju-lora-v2
```

**LoRA adapters** (`data/lora_adapters/`):
- `nju-v2/` — v2 with label masking fix
- `nju-v3/` — v3 (latest, `scripts/lora_train_v3.py`, `scripts/gen_lora_v3_answers.py`)

**GGUF models** (`data/models/`):
- `nju-lora-v3-Q4_K_M.gguf` — 4-bit quantized LoRA v3 for Ollama (4.7 GB)
- `bge-reranker-nju/` — fine-tuned reranker on NJU domain (2.2 GB)

**Merged models** (`data/models/`):
- `Qwen3-8B-NJU-LoRA-v3/` — 4-bit bnb merged weights for vLLM/transformers (6.0 GB)

## Eval system

- `data/eval/questions.csv` — 145 questions with `gold_source_ids` column (annotated via `scripts/annotate_gold_sources.py`). `should_refuse=true` for questions that are too personal/subjective to answer.
- `eval_rag.py` — end-to-end `/ask` evaluation (requires server). Reports latency, source coverage, keyword hit, refusal accuracy.
- `eval_retrieval.py` — direct retriever evaluation (recall@k, MRR, precision/recall). Supports `--rerank` and `--rewrite` flags.
- `eval_generation.py` — LLM-as-judge (faithfulness, relevance, refusal correctness, 1-5 scale). Uses DeepSeek as judge.
- `tune_weights.py` — grid search over BM25/Vector/Priority weight space (126 combos).
- `check_regression.py` — CI gate: compares 7 metrics against `*_baseline.json`, non-zero exit on regression.
- `error_taxonomy.py` — classifies F≤2 answers into 5 failure categories (retrieval, answer-ignores-retrieval, fact_check false positive, tier downgrade, judge noise).
- `coverage_gap_analysis.py` — per-topic metrics ranked by ROI priority for document acquisition.
- `data/eval/faithfulness_report.md` — detailed analysis of low-faithfulness answers, 5 hallucination patterns documented.

### Current eval metrics (v0.6.0)

| Metric | Value |
|--------|-------|
| End-to-end success | 144/144 (100%) |
| Has source ratio | 100% |
| Keyword hit ratio | 95.8% |
| Should-answer refused | 0 |
| Should-refuse answered | 0 |
| Avg latency | 2.23s |
| recall@5 (no rerank) | 0.831 |
| recall@5 (rerank) | 0.881 |
| MRR (rerank) | 0.612 |
| Context Precision@10 | 0.12 |
| Faithfulness | 2.99/5 |
| Relevance | 3.31/5 |
| Refusal Correctness | 4.83/5 |

## Key configuration

```bash
# LLM (local Qwen3-8B via Ollama)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1  # Must match OLLAMA_HOST port
LLM_MODEL=qwen3:8b-nothink

# LLM fallback
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# Retrieval — weights tuned via grid search on BGE-M3
BM25_TOP_K=10; VECTOR_TOP_K=10; HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2; HIGH_RISK_MIN_SCORE=0.25

# Features
ENABLE_RERANK=true           # cross-encoder re-ranker
ENABLE_QUERY_REWRITE=false   # colloquial→formal query normalization (default off)
ENABLE_CITATION_VERIFY=false # bigram overlap guardrail (default off)
ENABLE_LLM_FALLBACK=true     # DeepSeek fallback if Ollama fails
ENABLE_FACT_CHECK=true       # NER entity verification (Phase 1.1, default on)
ENABLE_HIGH_RISK_DEEPSEEK=true # route high-risk questions to DeepSeek
ENABLE_TWO_STAGE_GENERATION=false  # DEPRECATED — merged into system prompt

# Reranker
RERANKER_TYPE=cross_encoder  # cross_encoder | llm
LLM_RERANKER_BATCH_SIZE=15
LLM_RERANKER_FALLBACK_TO_CE=true  # fall back to cross-encoder if LLM reranker fails

# Confidence tiering (v0.6.3 — data-driven thresholds)
CONFIDENCE_TIER1_TOP1=0.70   # Tier 1: answer confidently
CONFIDENCE_TIER1_TOP3=0.55
CONFIDENCE_TIER3_TOP1=0.25   # Tier 3: skip LLM, direct referral

# Prompt budget (v0.5.1 — token-aware context trimming)
PROMPT_TOKEN_BUDGET=4096     # total prompt token budget
MAX_CHUNK_TOKENS=320         # per-chunk token cap (head+tail preserved)
MAX_CHUNKS_IN_PROMPT=6       # max chunks fed to LLM (was 12)

# LLM timeout & circuit breaker (v0.5.1)
LLM_REQUEST_TIMEOUT=20       # HTTP timeout for LLM requests (was 120)
LLM_TTFT_TIMEOUT_SECONDS=5   # stream first-token timeout

# GPU memory (v0.5.1)
EMPTY_CACHE_EVERY_N_REQUESTS=20  # periodic torch.cuda.empty_cache
EMPTY_CACHE_FREE_VRAM_MB=1500    # force cleanup when free VRAM < 1.5GB
RERANKER_DEVICE=auto             # auto | cuda | cpu (CPU mode saves ~1GB VRAM)

# Embedding
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

## Known issues (v0.6.0)

**Faithfulness 2.99/5** — significant improvement from 2.31 thanks to fact-check pipeline (NER entity verification removes or hedges fabricated numbers/dates). Remaining gap still driven by **Context Precision@10 = 0.12**: only ~1 in 8 retrieved chunks is relevant, so the LLM lacks anchor content on some questions.

**Context Precision@10 = 0.12** — the retrieval pipeline is noisy. 20 questions have recall@5 < 0.5. Some topics (社团/考研/体育课项目/校历) have only 1-2 source documents. Route A (retrieval improvement) plan exists but not yet executed.

**Reranker sigmoid discrimination is weak** — the cross-encoder outputs logits clustered near 0 (sigmoid ~0.5 for all chunks), providing minimal ranking signal. `scripts/finetune_reranker.py` can fine-tune on NJU-domain relevance pairs from gold-source annotations.

**LoRA v1 overfit** (F=0.0) — first LoRA attempt trained on all tokens including prompt, collapsed to empty outputs. v2 fixes this with label masking (prompt masked, only assistant tokens trained). v3 is the current production adapter.

## Deployment notes

**Startup** — one command:

```bash
./scripts/start_server.sh           # production
./scripts/start_server.sh --reload  # dev mode
```

The script preserves proxy vars (for HuggingFace/DeepSeek access), sets `NO_PROXY=localhost,127.0.0.1` to exclude local Ollama, enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, activates venv, and runs preflight checks before starting uvicorn.

**Preflight diagnostics** (run without starting server):

```bash
python scripts/preflight_check.py
```

Checks: CUDA, model weights, Ollama, VRAM, proxy vars (OK if set with NO_PROXY covering localhost), index files.

**Proxy setup**: WSL2 inherits Windows proxy settings. External access (HuggingFace, DeepSeek fallback) needs proxy; local Ollama must bypass it. `start_server.sh` handles this automatically — sets `NO_PROXY=localhost,127.0.0.1` and preserves existing proxy vars.

**Ollama server environment** (must be set where `ollama serve` runs):

```bash
# Source scripts/ollama_env.sh before starting ollama serve, or set manually:
export OLLAMA_FLASH_ATTENTION=1     # enable Flash Attention (Ada arch, CC 8.9)
export OLLAMA_KV_CACHE_TYPE=q8_0    # 8-bit KV cache (~50% VRAM savings)
export OLLAMA_KEEP_ALIVE=24h        # keep model loaded, avoid cold starts
```

Ollama version: **0.24.0** (stable; 0.12.0 had known long-context regression).

**Rebuild model** after Modelfile changes:

```bash
ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink
ollama show qwen3:8b-nothink --parameters  # verify num_ctx=8192, num_predict=400
```

**GPU memory**: 16GB is the minimum for Qwen3-8B + BGE-M3 + BGE-Reranker simultaneously. v0.5.1 mitigations:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation
- Periodic `torch.cuda.empty_cache()` every 20 requests (or when free < 1.5 GB)
- `RERANKER_DEVICE=cpu` frees ~1 GB VRAM if needed (reranker latency 200-500ms on CPU)
- Ollama `OLLAMA_KV_CACHE_TYPE=q8_0` saves ~0.5-1 GB

**Latency profile** (v0.5.1, with Modelfile and token budget):
- Normal questions (80%): 2-3s
- Long-context questions: 5-15s
- High-risk questions: ≤ 15s (template-based, not LLM-generated appendices)
- P99: ≤ 20s (down from 60-150s, thanks to num_predict=400 + stop sequences + token budget + 20s hard timeout → DeepSeek fallback)

**API endpoints**:
- `GET /health` — basic health check
- `GET /admin/health_deep` — full runtime snapshot: Ollama, GPU, models, index, cache
- `POST /ask` — non-streaming Q&A
- `POST /ask/stream` — SSE streaming Q&A
- `POST /feedback` — user feedback logging
- `POST /admin/ingest_url` — ingest a document by URL
- `GET /admin/staging` — list staged documents
- `GET /cache/stats` — LRU cache hit/miss statistics

**Thread safety**: GPU models (BGE-M3, BGE-Reranker) are NOT thread-safe. Calls to `.encode()` and `.predict()` are serialized via per-model locks. The LLM HTTP call to Ollama is outside the lock and can run concurrently.

**Docker** (for headless deployment without GPU):

```bash
docker compose up -d    # builds and starts on port 8000
```

The Docker image downloads a CPU embedding model at build time. GPU is not available inside the container, so vector retrieval will be slow — for production GPU use, run natively as described above.

## Design principles

- **Dependency injection**: `RAGPipeline` receives all components via constructor. No global state in pipeline logic.
- **Protocol interfaces**: `Retriever` and `Reranker` are `Protocol` types — any compatible object works.
- **Extensible**: Subclass `RiskClassifier` to add keywords; override `RAGPipeline` step methods to customize flow.
- **Graceful degradation**: Vector index missing → BM25 fallback. Primary LLM fails → fallback LLM. All new features have `.env` off-switches.
- **Backward compatibility**: Original `/ask` endpoint unchanged; new stream endpoint at `/ask/stream`. Module-level functions in `config.py` and `policy.py` preserved.

## Environment

- **OS**: Linux (WSL2 Ubuntu on Windows 11). WSL2 must use `networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`.
- **GPU**: NVIDIA RTX 4070 Ti Super 16GB, CUDA 12.4 driver. PyTorch must match: `torch==2.6.0+cu124`.
- **Ollama**: Runs natively in WSL, exposes OpenAI-compatible API at `localhost:11434/v1`.
- Scripts that import `app.*` need `PYTHONPATH=.` prefix.
- Never commit `.env` (it's in `.gitignore`).
- `docs/` directory contains reference documents: `dev_contract.md` (chunk/API format contract), `deployment_guide.md`, `requirement.md`, `risk_policy.md`, etc. The README is slightly outdated (v0.5.2); this CLAUDE.md reflects the current v0.6.3 state.
