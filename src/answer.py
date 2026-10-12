"""Retrieve context, ask the LLM, return an answer with citations."""
import re

from src import llm
from src.retrieve import retrieve

IDK = "I don't know based on the KNUST documents I have."

SYSTEM = f"""You answer questions from KNUST students using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Never use outside knowledge about KNUST or other universities.
- After each claim, cite the source number in square brackets, e.g. [2].
- If the sources do not contain the answer, reply exactly: "{IDK}"
- Be concise: 1-4 sentences, plain language."""


def format_sources(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        pages = ", ".join(map(str, c["pages"]))
        parts.append(f"[{i}] ({c['doc']}, p. {pages})\n{c['text']}")
    return "\n\n".join(parts)


def ask(question: str, k: int | None = None, mode: str | None = None, model: str | None = None) -> dict:
    chunks = retrieve(question, k=k, mode=mode)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"},
    ]
    out = llm.chat(messages, model=model)
    text = out["text"].strip()

    # Handles [2], [1, 4] and [1][3] styles
    nums = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", group)}
    cited_nums = sorted(n for n in nums if 0 < n <= len(chunks))
    citations = [
        {"n": n, "doc": chunks[n - 1]["doc"], "pages": chunks[n - 1]["pages"]} for n in cited_nums
    ]
    return {
        "question": question,
        "answer": text,
        "refused": text.startswith("I don't know"),
        "citations": citations,
        "retrieved": [{"doc": c["doc"], "pages": c["pages"], "score": round(c["score"], 4)} for c in chunks],
        "latency_s": out["latency_s"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
    }


if __name__ == "__main__":
    import json
    import sys
    q = " ".join(sys.argv[1:]) or "What happens if I miss an end-of-semester exam?"
    print(json.dumps(ask(q), indent=2))
