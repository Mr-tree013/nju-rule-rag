import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import APP_TITLE
from app.rag_pipeline import answer_question

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
        return {
            "question": req.question,
            "answer": "请输入您的问题。",
            "risk_level": "low",
            "need_human_confirm": False,
            "sources": [],
        }

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
