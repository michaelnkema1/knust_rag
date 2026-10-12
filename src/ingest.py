"""Parse PDFs in data/raw/, chunk them (keeping page numbers), embed into Chroma.

Usage:  python -m src.ingest            # rebuild the index
        CHUNK_SIZE=300 python -m src.ingest   # experiment with chunk size
"""
import json
import re
from pathlib import Path

from pypdf import PdfReader

from src import config


def clean(text: str) -> str:
    text = text.replace("­", "")              # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)         # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if text:
            pages.append({"doc": pdf_path.name, "page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Sliding window over words, across page boundaries.

    Each chunk records every page it touches, so citations stay accurate.
    """
    words = []  # (word, page)
    for p in pages:
        words.extend((w, p["page"]) for w in p["text"].split())

    chunks, start, step = [], 0, max(1, size - overlap)
    doc = pages[0]["doc"] if pages else ""
    while start < len(words):
        window = words[start:start + size]
        page_nums = sorted({pg for _, pg in window})
        chunks.append({
            "id": f"{doc}::{start}",
            "doc": doc,
            "pages": page_nums,
            "text": " ".join(w for w, _ in window),
        })
        if start + size >= len(words):
            break
        start += step
    return chunks


def build_chunks() -> list[dict]:
    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {config.RAW_DIR}. Download the source documents first.")
    all_chunks = []
    for pdf in pdfs:
        pages = read_pages(pdf)
        chunks = chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        print(f"{pdf.name}: {len(pages)} pages with text -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks


def save_chunks(chunks: list[dict]) -> None:
    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def index_chunks(chunks: list[dict]) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
    col = client.create_collection(config.COLLECTION, embedding_function=embed_fn,
                                   metadata={"hnsw:space": "cosine"})
    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        col.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            metadatas=[{"doc": c["doc"], "pages": ",".join(map(str, c["pages"]))} for c in part],
        )
    print(f"Indexed {len(chunks)} chunks into '{config.COLLECTION}' (chunk size {config.CHUNK_SIZE})")


if __name__ == "__main__":
    chunks = build_chunks()
    save_chunks(chunks)
    index_chunks(chunks)
