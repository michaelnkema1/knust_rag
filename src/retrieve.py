"""Vector, BM25 and hybrid retrieval over the indexed chunks."""
import json
import re
from functools import lru_cache

from src import config


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=1)
def _bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi([_tokenize(c["text"]) for c in _chunks()])


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    from chromadb.utils import embedding_functions
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    return client.get_collection(config.COLLECTION, embedding_function=embed_fn)


def vector_search(query: str, k: int) -> list[dict]:
    res = _collection().query(query_texts=[query], n_results=k)
    by_id = {c["id"]: c for c in _chunks()}
    out = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        c = dict(by_id[cid])
        c["score"] = 1 - dist
        out.append(c)
    return out


def bm25_search(query: str, k: int) -> list[dict]:
    scores = _bm25().get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    out = []
    for i in ranked:
        c = dict(_chunks()[i])
        c["score"] = float(scores[i])
        out.append(c)
    return out


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
