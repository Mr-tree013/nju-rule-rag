"""
FastAPI entry point for NJU Rule RAG Bot.

Provides GET /health and POST /ask endpoints.
"""

import logging
import re
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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


@app.get("/health")
def health():
    """Health check.  Returns basic status; retriever stats only if cached."""
    s = create_settings()
    warnings = s.validate()

    resp: dict = {
        "status": "ok",
        "version": "0.2.0",
        "chunks_file": s.chunks_file,
        "vector_enabled": s.enable_vector,
    }

    # Only include retriever status if the singleton is already loaded.
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


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise EmptyQuestionError()

    try:
        result = answer_question(req.question)
        logger.info(
            "question=%s risk=%s confirm=%s sources=%d latency=%.2f",
            req.question[:50],
            result["risk_level"],
            result["need_human_confirm"],
            len(result["sources"]),
            result["debug"].get("latency", 0),
        )
        return result
    except Exception as exc:
        logger.error(
            "question=%s error=%s",
            req.question[:50],
            str(exc)[:200],
        )
        return JSONResponse(
            status_code=500,
            content={
                "question": req.question,
                "answer": "系统暂时不可用，请稍后再试。",
                "risk_level": "unknown",
                "need_human_confirm": True,
                "sources": [],
                "error": "internal_error",
            },
        )


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

    text = _RE_CQ.sub("", raw).strip()

    # Only respond to /问 or /ask commands (ignore casual chat)
    has_cmd = "/问" in text or "/ask" in text
    if not has_cmd:
        return {"reply": ""}

    logger.info("qq_webhook processing text=%.200s", text)
    reply = handle_message(text)
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
