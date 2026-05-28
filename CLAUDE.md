# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NJU Rule RAG — a retrieval-augmented generation bot for Nanjing University undergraduate academic rules. Students ask questions in natural language; the system retrieves relevant regulatory documents and generates answers with source citations and risk-level classification.

**Current status**: v0.4.0. 70 source documents → 3771 chunks. Local Qwen3-8B + BGE-M3 embeddings + BGE-Reranker-v2-m3 two-stage retrieval. 120 tests pass. QQ Bot integration.

## Commands

```bash
source .venv/bin/activate

# Start dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest
pytest tests/test_pipeline.py -x

# ── Data pipeline ──

PYTHONPATH=. python scripts/build_chunks.py
PYTHONPATH=. python scripts/build_index.py
python scripts/validate_sources.py && python scripts/validate_chunks.py

# ── Evaluation ──

python scripts/eval_rag.py                  # 70-question /ask eval (needs server)
PYTHONPATH=. python scripts/eval_retrieval.py          # retrieval metrics (direct)
PYTHONPATH=. python scripts/eval_retrieval.py --rerank # with reranker
PYTHONPATH=. python scripts/eval_generation.py         # LLM-as-judge scoring
PYTHONPATH=. python scripts/tune_weights.py           # weight grid search
PYTHONPATH=. python scripts/check_regression.py       # CI regression gate
python scripts/annotate_gold_sources.py                # refresh gold-source labels

# ── Stream test ──
curl -N -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
  -d '{"question": "补考没过怎么办"}'
```

## Architecture

```
POST /ask {"question": "..."}
        │
        ▼
[QueryRewriter]        口语化规范化（should_rewrite()守卫，默认跳过正式问题）
        │
        ▼
TwoLayerRiskClassifier  L1关键词(高召回) → L2 embedding centroid相似度消歧
        │
        ▼
HybridRetriever        BM25(0.25) + BGE-M3 Vector(0.45) + Priority(0.30)
        │
        ▼
CrossEncoderReranker   BGE-Reranker-v2-m3 二阶段精排（40候选→12结果）
        │
        ▼
_filter → _dedup       score阈值过滤 → max 3/source, 12 total
        │
        ▼
LLM (Qwen3-8B)         fallback→DeepSeek on failure
        │
        ▼
[_verify_citations]    答案句bigram与来源重叠度检查（ENABLE_CITATION_VERIFY）
        │
        ▼
_format_response       长度截断 + 高风险通知(含部门联系方式) + 来源时效性
        │
        ▼
{ question, answer, risk_level, need_human_confirm, sources[], debug }
```

### Key module changes since v0.3.0

- **`app/reranker.py`** — `CrossEncoderReranker` (BGE-Reranker-v2-m3), Protocol `Reranker` interface.
- **`app/query_rewriter.py`** — `QueryRewriter` with `should_rewrite()` guard: only triggers on colloquial queries (≤6 chars or contains 咋办/挂科/能行吗 patterns).
- **`app/cache.py`** — `QACache`: LRU in-memory cache (200 entries, 1h TTL), thread-safe.
- **`app/policy.py`** — `TwoLayerRiskClassifier` extends `RiskClassifier` with BGE-M3 embedding centroid similarity for disambiguation. `high_risk_notice()` now accepts department contacts from source metadata.
- **`app/llm_client.py`** — Added `chat_stream()` for SSE token-by-token generation.
- **`app/retriever.py`** — `HybridRetriever.search()` now passes `top_k` through to sub-retrievers (was hardcoded to 10). `VectorRetriever` exposes `embedding_model` property. nju-life QA penalty reduced from 0.65→0.85.
- **`app/pipeline.py`** — Steps added: `_rewrite_query`, `_rerank`, `_generate_stream`, `_verify_citations`. `_build_prompt` accepts `is_process` for step-by-step formatting. `preload_pipeline()` runs a warmup query to pre-load all models into GPU. `_extract_sources` now includes `fetched_at`.

## Model inventory

| Model | Size | Where | Purpose |
|-------|------|-------|---------|
| Qwen3-8B (no-think) | 5.2 GB | Ollama `qwen3:8b-nothink` | LLM generation |
| BGE-M3 | 2.2 GB | sentence-transformers | Query/document embedding (1024-dim) |
| BGE-Reranker-v2-m3 | 1.0 GB | sentence-transformers | Cross-encoder reranking |

Total GPU memory: ~8-10 GB. Create the no-think variant via `ollama create qwen3:8b-nothink -f scripts/modelfile.qwen3-nothink`.

## Data pipeline

1. `data/sources.csv` — 70 source documents (priority 1-5, department, scope).
2. `scripts/build_chunks.py` — `data/processed/*.md` → `data/chunks/chunks.jsonl` (3771 chunks). Splits by article headings (including `**第X条**` bold markdown). 0 too-long chunks enforced via `_split_by_fixed_size` fallback.
3. `scripts/build_index.py` — BM25 (jieba) + Chroma (BGE-M3, 1024-dim). GPU auto-detection. `batch_size=8` for 16GB VRAM.

To add a document: `.md` → `data/processed/`, add row to `data/sources.csv`, then `build_chunks.py && build_index.py && validate_*`.

## Eval system

- `data/eval/questions.csv` — 70 questions with `gold_source_ids` column (annotated via `scripts/annotate_gold_sources.py`).
- `eval_rag.py` — end-to-end `/ask` evaluation (requires server).
- `eval_retrieval.py` — direct retriever evaluation (recall@k, MRR, precision/recall). Supports `--rerank` and `--rewrite` flags.
- `eval_generation.py` — LLM-as-judge (faithfulness, relevance, refusal correctness, 1-5 scale).
- `tune_weights.py` — grid search over BM25/Vector/Priority weight space (126 combos).
- `check_regression.py` — CI gate: compares 7 metrics against `*_baseline.json`, non-zero exit on regression.

## Key configuration

```bash
# LLM (local Qwen3-8B via Ollama)
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b-nothink

# LLM fallback
ENABLE_LLM_FALLBACK=true
FALLBACK_LLM_BASE_URL=https://api.deepseek.com
FALLBACK_LLM_MODEL=deepseek-chat

# Retrieval — weights tuned via grid search on BGE-M3
BM25_TOP_K=10; VECTOR_TOP_K=10; HYBRID_TOP_K=5
MIN_RELIABLE_SCORE=0.2; HIGH_RISK_MIN_SCORE=0.25

# Features (all default-off in .env.example)
ENABLE_RERANK=true
ENABLE_QUERY_REWRITE=true
ENABLE_CITATION_VERIFY=false
ENABLE_LLM_FALLBACK=true

# Embedding
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
```

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
