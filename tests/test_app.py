"""API tests. The LLM is mocked, so these run offline and cost nothing."""
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import app.main as m
from src import config
from src.answer import excerpt

FAKE = {"question": "q", "answer": "A [1].", "refused": False,
        "citations": [{"n": 1, "doc": "students_guide_2023.pdf", "pages": [78], "excerpt": "x"}],
        "retrieved": [], "latency_s": 1, "prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")  # no local PDFs -> redirect path
    m._recent.clear()
    with mock.patch.object(m, "ask", return_value=FAKE):
        yield TestClient(m.app)


def test_health_and_page(client):
    assert client.get("/health").json() == {"ok": True}
    assert "KNUST Regulations Assistant" in client.get("/").text


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hello there"}).status_code == 200
    assert client.post("/ask", json={"question": "x"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422


def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(config, "ASK_RATE_LIMIT", 2)
    codes = [client.post("/ask", json={"question": "hello there"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_pdf_redirects_to_official_copy_and_blocks_other_paths(client):
    r = client.get("/docs-pdf/students_guide_2023.pdf", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("https://www.knust.edu.gh/")
    assert client.get("/docs-pdf/..%2F.env").status_code == 404
    assert client.get("/docs-pdf/unknown.pdf").status_code == 404


def test_feedback_is_logged_and_validated(client):
    ok = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "down", "comment": "wrong page"})
    assert ok.json() == {"ok": True}
    assert client.post("/feedback", json={"question": "q", "answer": "a", "rating": "meh"}).status_code == 422
    row = json.loads(m.FEEDBACK_LOG.read_text().splitlines()[0])
    assert row["rating"] == "down" and row["comment"] == "wrong page"


def test_excerpt_picks_window_matching_the_answer():
    text = " ".join(f"w{i}" for i in range(300)) + " the Counselling Centre is in the Harper Building " + \
           " ".join(f"z{i}" for i in range(300))
    out = excerpt(text, "The Counselling Centre is located in the Harper Building")
    assert "Harper Building" in out and out.startswith("…") and out.endswith("…")
