"""FastAPI app: chat page, POST /ask, source PDFs, and answer feedback.

Run:  uvicorn app.main:app --reload
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask_endpoint(q: Question):
    try:
        return ask(q.question)
    except FileNotFoundError:
        raise HTTPException(503, "Index not built yet. Run: python -m src.ingest")


@app.get("/docs-pdf/{name}")
def source_pdf(name: str):
    """Serve an indexed source PDF so citations can link to it (e.g. /docs-pdf/x.pdf#page=78)."""
    allowed = {p.name: p for p in config.RAW_DIR.glob("*.pdf")}  # only files we indexed; blocks ../ paths
    if name not in allowed:
        raise HTTPException(404, "Document not found")
    return FileResponse(allowed[name], media_type="application/pdf")


@app.post("/feedback")
def feedback(fb: Feedback):
    """Append a thumbs up/down to data/feedback.jsonl. Thumbs-down rows are candidate eval questions."""
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **fb.model_dump(include={"question", "answer", "rating", "comment"}),
              "citations": [{"doc": c.get("doc"), "pages": c.get("pages")} for c in fb.citations]}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
