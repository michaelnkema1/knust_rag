"""Vector, BM25 and hybrid retrieval over the indexed chunks.

The index is two small files per chunk size, built by `python -m src.ingest`:
  data/index/chunks_<size>.jsonl     chunk text + doc + pages
  data/index/embeddings_<size>.npy   one normalized vector per chunk (same order)
Vector search is a dot product in NumPy, which is plenty fast for a few hundred chunks
and needs no vector database.
"""
import json
import re
from functools import lru_cache

import numpy as np

from src import config
from src.embed import embed_query


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def _embeddings() -> np.ndarray:
    emb = np.load(config.EMBEDDINGS_PATH)
    if len(emb) != len(_chunks()):
        raise RuntimeError(f"{config.EMBEDDINGS_PATH.name} and {config.CHUNKS_PATH.name} are out of sync; re-run ingest")
    return emb


@lru_cache(maxsize=1)
def _bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi([_tokenize(c["text"]) for c in _chunks()])


def vector_search(query: str, k: int) -> list[dict]:
    scores = _embeddings() @ embed_query(query)
    top = np.argsort(-scores)[:k]
    return [{**_chunks()[i], "score": float(scores[i])} for i in top]


def bm25_search(query: str, k: int) -> list[dict]:
    scores = _bm25().get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**_chunks()[i], "score": float(scores[i])} for i in ranked]


def hybrid_search(query: str, k: int, rrf_k: int = 60) -> list[dict]:
    """Reciprocal rank fusion of vector and BM25 results."""
    fused, keep = {}, {}
    for results in (vector_search(query, k * 2), bm25_search(query, k * 2)):
        for rank, c in enumerate(results):
            fused[c["id"]] = fused.get(c["id"], 0) + 1 / (rrf_k + rank + 1)
            keep[c["id"]] = c
    ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**keep[i], "score": fused[i]} for i in ids]


def retrieve(query: str, k: int | None = None, mode: str | None = None) -> list[dict]:
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    if mode == "bm25":
        return bm25_search(query, k)
    if mode == "hybrid":
        return hybrid_search(query, k)
    return vector_search(query, k)
