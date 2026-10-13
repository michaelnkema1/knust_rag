"""FastAPI app: chat page, POST /ask, source PDFs, and answer feedback.

Run:  uvicorn app.main:app --reload
"""
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from src import config
from src.answer import ask

app = FastAPI(title="KNUST Regulations Assistant")
STATIC = Path(__file__).parent / "static"
FEEDBACK_LOG = config.ROOT / "data" / "feedback.jsonl"


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Feedback(BaseModel):
    question: str = Field(..., max_length=500)
    answer: str = Field(..., max_length=5000)
    rating: Literal["up", "down"]
    comment: str = Field("", max_length=1000)
    citations: list[dict] = Field(default_factory=list)


# Simple in-memory rate limit per client IP, so a public link can't burn through the LLM quota.
_recent: dict[str, deque] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the real client is the first address in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _check_rate_limit(request: Request) -> None:
    limit = config.ASK_RATE_LIMIT
    if limit <= 0:
        return
    now, q = time.monotonic(), _recent[_client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many questions in a short time. Please wait a minute and try again.")
    q.append(now)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask_endpoint(q: Question, request: Request):
    _check_rate_limit(request)
    try:
        return ask(q.question)
    except FileNotFoundError:
        raise HTTPException(503, "Index not built yet. Run: python -m src.ingest")


@app.get("/docs-pdf/{name}")
def source_pdf(name: str):
    """Serve an indexed source PDF so citations can link to it (e.g. /docs-pdf/x.pdf#page=78).

    If the PDF isn't on this server (it isn't committed to git), redirect to the official public copy;
    browsers keep the #page=N part across the redirect.
    """
    local = {p.name: p for p in config.RAW_DIR.glob("*.pdf")}  # only files we indexed; blocks ../ paths
    if name in local:
        return FileResponse(local[name], media_type="application/pdf")
    if name in config.SOURCE_URLS:
        return RedirectResponse(config.SOURCE_URLS[name], status_code=307)
    raise HTTPException(404, "Document not found")


@app.post("/feedback")
def feedback(fb: Feedback):
    """Append a thumbs up/down to data/feedback.jsonl. Thumbs-down rows are candidate eval questions.

    Note: on hosts with an ephemeral disk (e.g. Render free tier) this file is lost on restart.
    """
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **fb.model_dump(include={"question", "answer", "rating", "comment"}),
              "citations": [{"doc": c.get("doc"), "pages": c.get("pages")} for c in fb.citations]}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
