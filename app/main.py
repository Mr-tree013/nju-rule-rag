from fastapi import FastAPI
from pydantic import BaseModel

from app.config import APP_TITLE

app = FastAPI(title=APP_TITLE)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    return {
        "question": req.question,
        "answer": "RAG pipeline is not ready yet.",
        "risk_level": "low",
        "need_human_confirm": False,
        "sources": [],
    }
