from src.ingest import chunk_pages, clean


def test_clean_rejoins_hyphenated_words():
    assert clean("regu-\nlations") == "regulations"


def test_chunks_track_pages_and_overlap():
    pages = [
        {"doc": "d.pdf", "page": 1, "text": " ".join(f"a{i}" for i in range(10))},
        {"doc": "d.pdf", "page": 2, "text": " ".join(f"b{i}" for i in range(10))},
    ]
    chunks = chunk_pages(pages, size=8, overlap=2)
    assert chunks[0]["pages"] == [1]
    assert any(c["pages"] == [1, 2] for c in chunks)       # a chunk spans the page break
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]  # overlap
    assert chunks[-1]["text"].split()[-1] == "b9"          # nothing dropped at the end
