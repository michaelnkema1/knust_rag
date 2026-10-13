"""Text embeddings with fastembed (ONNX Runtime, no PyTorch), so the app fits small servers.

BAAI/bge-small-en-v1.5 here gives the same vectors as the sentence-transformers version
(cosine 1.00000 on all indexed chunks), so earlier eval results still apply.

  python -m src.embed     # download the model into data/models (used as a build step on Render)
"""
from functools import lru_cache

import numpy as np

from src import config


@lru_cache(maxsize=1)
def model():
    from fastembed import TextEmbedding
    return TextEmbedding(config.EMBED_MODEL, cache_dir=str(config.MODEL_DIR), threads=config.EMBED_THREADS)


def _normalize(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def embed_documents(texts: list[str]) -> np.ndarray:
    # Small batches keep memory low; large batches of long chunks can use several GB.
    return _normalize(np.array(list(model().embed(texts, batch_size=8))))


def embed_query(text: str) -> np.ndarray:
    return _normalize(np.array(list(model().query_embed(text)))[0])


if __name__ == "__main__":
    print("embedding dim:", embed_query("warm-up").shape[0], "| model cached in", config.MODEL_DIR)
