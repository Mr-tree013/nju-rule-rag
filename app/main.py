"""
FastAPI entry point for NJU Rule RAG Bot.

Provides GET /health and POST /ask endpoints.
"""

import logging
import os
import re
import sys
import time
import uuid

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path

from app.cache import qa_cache, query_cache
from app.config import APP_TITLE, create_settings
from app.errors import EmptyQuestionError
from app.pipeline import answer_question, preload_pipeline
from app.qq_bot import handle_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title=APP_TITLE)

# ── CORS (allows browser-based frontends) ────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    """Preload the RAG pipeline so the first request isn't slow."""
    logger.info("Preloading RAG pipeline...")
    preload_pipeline()
    logger.info("Startup complete.")


class AskRequest(BaseModel):
    question: str


def _log_request(request_id: str, question: str, result: dict):
    """Log ask request details for feedback correlation."""
    log_dir = Path("data/requests")
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "request_id": request_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": result.get("answer", "")[:500],
        "risk_level": result.get("risk_level", ""),
        "tier": result.get("debug", {}).get("confidence_tier", "?"),
        "fact_check": result.get("debug", {}).get("fact_check", {}),
        "sources": result.get("sources", []),
        "latency": result.get("debug", {}).get("latency", 0),
    }
    fname = time.strftime("%Y%m") + ".jsonl"
    try:
        with open(log_dir / fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _log_feedback_entry(rating: str, question: str, answer: str = "",
                        request_id: str = "", comment: str = "",
                        sources: str = "", tier: str = "",
                        fact_check: dict | None = None):
    """Write one feedback entry. Shared by /feedback endpoint and QQ bot."""
    log_dir = Path("data/feedback")
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer[:500],
        "rating": rating,
        "comment": comment,
        "sources": sources,
        "request_id": request_id,
        "tier": tier,
        "fact_check": fact_check or {},
    }
    fname = time.strftime("%Y-%m-%d") + ".jsonl"
    try:
        with open(log_dir / fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise EmptyQuestionError()

    # Check cache
    cached = qa_cache.get(req.question)
    if cached is not None:
        cached["debug"]["cached"] = True
        if "request_id" not in cached:
            cached["request_id"] = str(uuid.uuid4())[:8]
        return cached

    request_id = str(uuid.uuid4())[:8]
    try:
        result = answer_question(req.question)
        result["request_id"] = request_id
        result["debug"]["cached"] = False
        qa_cache.set(req.question, result)
        _log_request(request_id, req.question, result)
        logger.info(
            "request_id=%s question=%s risk=%s confirm=%s sources=%d latency=%.2f",
            request_id,
            req.question[:50],
            result["risk_level"],
            result["need_human_confirm"],
            len(result["sources"]),
            result["debug"].get("latency", 0),
        )
        return result
    except Exception as exc:
        logger.error(
            "request_id=%s question=%s error=%s",
            request_id,
            req.question[:50],
            str(exc)[:200],
        )
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "question": req.question,
                "answer": "系统暂时不可用，请稍后再试。",
                "risk_level": "unknown",
                "need_human_confirm": True,
                "sources": [],
                "error": "internal_error",
            },
        )


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Streaming variant of /ask — SSE token-by-token generation."""
    if not req.question or not req.question.strip():
        return JSONResponse(status_code=400, content={"error": "Empty question"})

    from app.pipeline import _get_pipeline

    pipeline = _get_pipeline()
    t_start = time.time()

    # 1. Classify
    classification = pipeline._classify(req.question)

    # 2. Rewrite query (optional)
    search_query = req.question
    if pipeline._query_rewriter and pipeline._settings.enable_query_rewrite:
        search_query = pipeline._rewrite_query(req.question)

    # 3. Retrieve
    try:
        if pipeline._reranker and pipeline._settings.enable_rerank:
            chunks = pipeline._retrieve(search_query, top_k=pipeline._settings.rerank_candidate_k)
        else:
            chunks = pipeline._retrieve(search_query)
    except Exception:
        chunks = []

    retrieval_count = len(chunks)

    # 4. Rerank
    if pipeline._reranker and pipeline._settings.enable_rerank and chunks:
        chunks = pipeline._rerank(search_query, chunks)

    # 5. Filter & dedup & decide tier
    reliable = pipeline._filter_chunks(chunks, classification.level)
    top_chunks = pipeline._dedup_chunks(reliable) if reliable else []

    if not top_chunks:
        result = pipeline._no_evidence_response(
            req.question, classification, t_start, retrieval_count
        )
        return JSONResponse(content=result)

    confidence_tier, tier_top1, tier_top3 = pipeline._decide_confidence_tier(top_chunks)

    # Tier 3: direct referral, skip LLM
    if confidence_tier == "3":
        result = pipeline._tier3_response(
            req.question, classification, t_start, retrieval_count, {},
            {"top1": tier_top1, "top3_avg": tier_top3},
        )
        return JSONResponse(content=result)

    # 6. Build prompt (with tier-specific instructions)
    messages, prompt_tokens, prompt_chunks = pipeline._build_prompt(
        req.question, top_chunks, classification.level, classification.is_process,
        confidence_tier=confidence_tier,
    )

    async def generate():
        try:
            # Phase 1: retrieval done, starting generation
            yield f"data: {json.dumps({'phase': 'generating', 'chunks': retrieval_count})}\n\n"

            full_answer = ""
            for token in pipeline._generate_stream(messages):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
                await asyncio.sleep(0)

            # Phase 2: fact_check post-processing
            answer_text = full_answer
            fact_check_debug = {}
            if pipeline._settings.enable_fact_check:
                from app.fact_check import apply_fact_check
                fc_result = apply_fact_check(full_answer, top_chunks, confidence_tier)
                answer_text = fc_result["answer"]
                fact_check_debug = fc_result["debug"]
                if answer_text != full_answer:
                    yield f"data: {json.dumps({'phase': 'corrected', 'answer': answer_text, 'tier': fc_result.get('tier', confidence_tier)})}\n\n"
                    confidence_tier = fc_result.get("tier", confidence_tier)

            # Phase 3: format final response
            result = pipeline._format_response(
                req.question, answer_text, classification, top_chunks,
                t_start, retrieval_count,
                prompt_tokens=prompt_tokens, prompt_chunks=prompt_chunks,
                confidence_tier=confidence_tier,
                tier_top1=tier_top1, tier_top3=tier_top3,
                fact_check_debug=fact_check_debug,
            )
            yield f"data: {json.dumps({'done': True, 'result': result})}\n\n"
        except Exception as exc:
            logger.error("stream error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)[:200]})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class FeedbackRequest(BaseModel):
    question: str
    answer: str = ""
    rating: str = ""  # "up" or "down"
    comment: str = ""
    sources: str = ""  # JSON-serialized source chunks (for training pair construction)
    request_id: str = ""  # correlate with ask log for full debug info


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Log user feedback for eval set improvement and training data accumulation.

    Each down-vote with sources can be converted to a training pair:
      query=question, positive=gold_source (from manual label), negative=retrieved
    """
    # If request_id provided, try to load the original request debug info
    tier = ""
    fact_check = None
    if req.request_id:
        req_log_dir = Path("data/requests")
        fname = time.strftime("%Y%m") + ".jsonl"
        req_log = req_log_dir / fname
        if req_log.exists():
            try:
                with open(req_log, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            if r.get("request_id") == req.request_id:
                                tier = r.get("tier", "")
                                fact_check = r.get("fact_check", {})
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

    _log_feedback_entry(
        rating=req.rating,
        question=req.question,
        answer=req.answer,
        request_id=req.request_id,
        comment=req.comment,
        sources=req.sources,
        tier=tier,
        fact_check=fact_check,
    )
    logger.info("feedback rating=%s question=%.50s", req.rating, req.question)
    return {"status": "ok", "message": "感谢反馈！"}


@app.get("/cache/stats")
def cache_stats():
    return {
        "memory": qa_cache.stats(),
        "persistent": query_cache.stats(),
    }


@app.get("/health")
def health_basic():
    """Basic health check — fast, lightweight."""
    s = create_settings()
    warnings = s.validate()
    resp: dict = {
        "status": "ok",
        "version": "0.5.1",
        "chunks_file": s.chunks_file,
        "vector_enabled": s.enable_vector,
    }
    try:
        from app.pipeline import _pipeline
        if _pipeline is not None:
            status = _pipeline._retriever.status()
            resp["retriever"] = {
                "bm25_loaded": status.get("bm25_loaded", False),
                "bm25_chunks": status.get("bm25_chunks", 0),
                "vector_loaded": status.get("vector_loaded", False),
            }
    except Exception:
        pass
    if warnings:
        resp["config_warnings"] = warnings
    return resp


@app.get("/admin/health_deep")
def health_deep():
    """Deep health check — full runtime snapshot including GPU and Ollama."""
    from app.health import get_deep_health
    from app.config import create_settings
    s = create_settings()
    from app.cache import qa_cache, query_cache
    result = get_deep_health(s.project_root, cache_stats_fn=qa_cache.stats)
    # Phase 1.1 — expose fact check status
    result["fact_check_enabled"] = s.enable_fact_check
    return result


# ── Admin: URL ingestion endpoint (F.4) ──────────────────────────────


class IngestRequest(BaseModel):
    url: str
    source_id: str = ""
    title: str = ""
    department: str = ""
    scope: str = "本科生"
    priority: int = 3


@app.post("/admin/ingest_url")
def ingest_url(req: IngestRequest):
    """Submit a URL for ingestion into the corpus.

    Fetches the page, converts to markdown, saves to data/staging/,
    and returns a review reference for the CLI tool.
    """
    import os
    from pathlib import Path

    from app.crawl.fetcher import fetch_and_stage

    staging_dir = Path(os.getenv("STAGING_DIR", "data/staging"))
    result = fetch_and_stage(req.url, staging_dir=staging_dir, source_id=req.source_id or None)

    if result["status"] == "staged":
        return {
            "status": "staged",
            "file": result["file_path"],
            "content_hash": result["content_hash"],
            "content_length": result.get("content_length", 0),
            "message": (
                f"文档已暂存。运行 python scripts/review_staging.py "
                f"审核后入库。"
            ),
        }
    else:
        return {
            "status": "error",
            "error": result.get("error", "unknown"),
            "message": "抓取失败，请检查 URL 是否可访问。",
        }


@app.get("/admin/staging")
def list_staging():
    """List all staged documents waiting for review."""
    from pathlib import Path
    import os

    staging_dir = Path(os.getenv("STAGING_DIR", "data/staging"))
    if not staging_dir.exists():
        return {"staged": []}

    items = []
    for f in sorted(staging_dir.glob("*.md"), reverse=True):
        stat = f.stat()
        items.append({
            "file": str(f),
            "size": stat.st_size,
            "modified": stat.st_mtime,
        })
    return {"staged": items[:20], "total": len(items)}


# ── QQ Bot webhook (OneBot v11 HTTP 回调) ────────────────────────────

_RE_CQ = re.compile(r"\[CQ:\w+,.*?\]")


@app.post("/qq")
def qq_webhook(data: dict):
    t0 = time.time()
    logger.info("qq_webhook keys=%s raw_message=%.200s", list(data.keys()), data.get("raw_message", ""))

    if data.get("message_type") != "group":
        logger.info("qq_webhook skip: not group message")
        return {"reply": ""}

    raw = data.get("raw_message", "")
    group_id = data.get("group_id", "")
    user_id = data.get("user_id", "")

    # ── Feedback detection (before @mention check) ────────────────
    from app.qq_bot import check_feedback
    fb = check_feedback(raw, group_id, user_id)
    if fb:
        _log_feedback_entry(
            rating=fb["rating"],
            question=fb["question"],
            request_id=fb["request_id"],
        )
        logger.info("qq_webhook feedback rating=%s user=%s", fb["rating"], user_id)
        return {"reply": "感谢反馈！"}

    # Only respond when @mentioned
    self_id = str(create_settings().qq_bot_self_id)
    is_mentioned = f"[CQ:at,qq={self_id}]" in raw
    if not is_mentioned and "/ask" not in raw and "/问" not in raw:
        return {"reply": ""}

    text = _RE_CQ.sub("", raw).strip()
    if not text:
        return {"reply": ""}

    logger.info("qq_webhook processing text=%.200s", text)
    from app.qq_bot import format_reply_from_data, get_history, add_to_history

    # Inject conversation history so follow-up questions get context
    conv_history = get_history(group_id, user_id)
    result = answer_question(text, conversation_history=conv_history)
    request_id = str(uuid.uuid4())[:8]
    _log_request(request_id, text, result)
    reply = format_reply_from_data(text, result,
                                   group_id=group_id, user_id=user_id,
                                   request_id=request_id)
    if reply:
        add_to_history(group_id, user_id, text,
                       result.get("answer", "") if result else "")
    if not reply:
        return {"reply": ""}
    logger.info("qq_webhook reply=%.100s elapsed=%.1fs", reply, time.time() - t0)
    return {"reply": reply}


@app.exception_handler(EmptyQuestionError)
def handle_empty_question(request: Request, exc: EmptyQuestionError):
    return JSONResponse(
        status_code=400,
        content={
            "question": "",
            "answer": "请输入您的问题。",
            "risk_level": "low",
            "need_human_confirm": False,
            "sources": [],
        },
    )
