"""FastAPI app: POST /ask and a chat page at /.

Run:  uvicorn app.main:app --reload
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.answer import ask

app = FastAPI(title="KNUST Regulations Assistant")
STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


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
