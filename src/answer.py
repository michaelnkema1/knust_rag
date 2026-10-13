"""Retrieve context, ask the LLM, return an answer with citations."""
import re

from src import config, llm
from src.retrieve import retrieve

IDK = "I don't know based on the KNUST documents I have."

# Prompt versions are kept side by side so eval runs are reproducible; pick one with PROMPT_VERSION in .env.
SYSTEM_PROMPTS = {
    # v1 (baseline): strict. Eval showed it over-refuses: 8/79 answerable questions got "I don't know",
    # 4 of them even though the right page had been retrieved.
    "v1": f"""You answer questions from KNUST students using ONLY the numbered sources provided.
Rules:
- Use only facts stated in the sources. Never use outside knowledge about KNUST or other universities.
- After each claim, cite the source number in square brackets, e.g. [2].
- If the sources do not contain the answer, reply exactly: "{IDK}"
- Be concise: 1-4 sentences, plain language.""",

    # v2: same grounding rules, but answers whenever the sources support an answer, even partially
    # or in different wording, and refuses only when nothing in the sources is relevant.
    "v2": f"""You answer questions from KNUST students using ONLY the numbered sources provided.
Students write casually and may use different words from the official text, so read every source carefully
and match the question to the relevant rule, figure or procedure even if the wording differs.
Rules:
- Use only facts stated in the sources. Never use outside knowledge about KNUST or other universities.
- If any source answers the question, fully or in part, give that answer. If only part is covered,
  answer that part and say briefly which detail the documents don't cover.
- Include specific figures, grades, durations and requirements exactly as the sources state them.
- After each claim, cite the source number in square brackets, e.g. [2].
- Only if none of the sources contain information relevant to the question, reply exactly: "{IDK}"
- Be concise: 1-4 sentences, plain language.""",
}
SYSTEM = SYSTEM_PROMPTS[config.PROMPT_VERSION]


def excerpt(text: str, answer: str = "", words: int = 110) -> str:
    """Pick the ~110-word window of a chunk that shares the most words with the answer, for display."""
    toks = text.split()
    if len(toks) <= words:
        return " ".join(toks)
    key = {w for w in re.findall(r"[a-z]{4,}", answer.lower())}
    hits = [1 if re.sub(r"[^a-z]", "", t.lower()) in key else 0 for t in toks]
    cur = sum(hits[:words])
    best, score = 0, cur
    for i in range(1, len(toks) - words + 1):
        cur += hits[i + words - 1] - hits[i - 1]
        if cur > score:
            best, score = i, cur
    out = " ".join(toks[best:best + words])
    return ("…" if best > 0 else "") + out + ("…" if best + words < len(toks) else "")


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
        {"n": n, "doc": chunks[n - 1]["doc"], "pages": chunks[n - 1]["pages"],
         "excerpt": excerpt(chunks[n - 1]["text"], text)}
        for n in cited_nums
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
