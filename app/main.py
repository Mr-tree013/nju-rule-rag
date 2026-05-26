"""
FastAPI entry point for NJU Rule RAG Bot.

Provides GET /health and POST /ask endpoints.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import APP_TITLE
from app.errors import EmptyQuestionError
from app.pipeline import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

app = FastAPI(title=APP_TITLE)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


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
