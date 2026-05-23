# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NJU Rule RAG — a retrieval-augmented generation bot for Nanjing University undergraduate academic rules and procedures. Students ask questions in natural language; the system retrieves relevant regulatory documents and generates answers with source citations and risk-level classification.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Start dev server (hot reload)
uvicorn app.main:app --reload

# Run tests
pytest

# Run a single test file
pytest tests/test_answer_policy.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

The system follows a standard RAG pipeline:

```
Question → HybridRetriever (BM25 + vector) → context chunks → LLM → answer
                                                              ↓
                                                     AnswerPolicy (risk classify)
```

Key layers:

- **`app/main.py`** — FastAPI entry point. `/health` for liveness, `/ask` for the RAG endpoint.
- **`app/retriever.py`** — Retrieval logic: BM25Retriever (keyword exact match via `rank-bm25`), ChromaVectorRetriever (semantic search), and a `hybrid_search` that merges results with priority-weighted scoring.
- **`app/rag_pipeline.py`** — Orchestrates retrieval → prompt construction → LLM call → response formatting.
- **`app/answer_policy.py`** — Keyword-based risk classification (low/medium/high), process-question detection, and refusal/no-evidence response templates.
- **`app/llm_client.py`** — LLM API wrapper (embeddings + chat), provider-agnostic via env config.
- **`app/config.py`** — Reads all settings from `.env` via `python-dotenv`.
- **`scripts/`** — Offline data pipeline: crawl → parse → chunk → index → eval.

## Data flow

1. `data/sources.csv` lists all document sources with metadata (priority 1-5).
2. `scripts/crawl_sources.py` downloads raw HTML/PDF to `data/raw/`, each with a `.metadata.json` sidecar.
3. `scripts/parse_documents.py` extracts clean text to `data/processed/{source_id}.md`.
4. `scripts/build_chunks.py` splits text into chunks by article headings (`第X条`, `一、`, etc.) and writes `data/chunks/chunks.jsonl`.
5. `scripts/build_index.py` builds BM25 and Chroma vector indices in `data/index/`.
6. At query time, `rag_pipeline.answer_question()` runs hybrid retrieval, constructs a prompt from chunks, calls the LLM, applies risk policy, and returns `{answer, risk_level, sources, need_human_confirm}`.

## Risk policy

- **low**: general system inquiries — answer normally with sources.
- **medium**: involves personal situations (补考, 重修, 转专业) — answer general rules, flag `need_human_confirm=true`.
- **high**: 退学, 处分, 作弊, 学位, 毕业资格 — explain general rules only, never give personal conclusions, strongly advise contacting an academic advisor.

When no relevant chunks are found, the system returns a standard refusal response rather than guessing.

## Source priorities

Priority 1 (highest): current official university-level documents. Priority 5 (lowest): student handbook summaries. See `docs/source_priority.md` for the full table. The hybrid retriever weights priority in its final score.

## Environment

Copy `.env.example` to `.env` and fill in LLM/embedding API keys. Never commit `.env` or API keys.
