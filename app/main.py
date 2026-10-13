"""FastAPI app: chat page, POST /ask, source PDFs, and answer feedback.

Run:  uvicorn app.main:app --reload
"""
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from src import config
from src.answer import ask

log = logging.getLogger("knust_rag")
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
    # Turn LLM provider failures into clear messages (and log the real cause for the server logs).
    except openai.RateLimitError as e:
        log.warning("LLM rate limit: %s", e)
        raise HTTPException(503, "The AI service is busy or its free daily limit has been reached. "
                                 "Please try again in a few minutes.")
    except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
        log.error("LLM auth/permission error: %s", e)
        raise HTTPException(503, "The AI service rejected this server's credentials or location. "
                                 "The site owner needs to check the API key settings.")
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        log.error("LLM connection error: %s", e)
        raise HTTPException(504, "Couldn't reach the AI service. Please try again.")
    except openai.APIStatusError as e:
        log.error("LLM API error %s: %s", e.status_code, e)
        raise HTTPException(502, f"The AI service returned an error ({e.status_code}). Please try again.")


@app.get("/debug/llm")
def debug_llm(request: Request):
    """Owner diagnostic: one tiny LLM call, returning the error type if it fails (never the key).
    Shares the /ask rate limit so it can't be used to drain the quota."""
    _check_rate_limit(request)
    from src import llm
    try:
        out = llm.chat([{"role": "user", "content": "Reply with OK"}], max_retries=0)
        return {"ok": True, "model": config.LLM_MODEL, "reply": out["text"][:20], "latency_s": out["latency_s"]}
    except Exception as e:  # noqa: BLE001 - diagnostic endpoint
        import hashlib
        key = config.LLM_API_KEY.strip()
        return {"ok": False, "model": config.LLM_MODEL, "base_url": config.LLM_BASE_URL,
                "key_set": bool(key), "key_len": len(key),
                # A short hash lets the owner check which key the server has without revealing it.
                "key_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:10],
                "key_has_whitespace": key != config.LLM_API_KEY,
                "error": type(e).__name__, "detail": str(e)[:1500]}


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
