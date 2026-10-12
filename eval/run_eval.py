"""Run the eval set and write per-question results + a summary.

Usage:
  python -m eval.run_eval                       # current config
  RETRIEVAL_MODE=hybrid python -m eval.run_eval --tag hybrid
  python -m eval.run_eval --retrieval-only      # no LLM calls, just hit@k

Metrics:
  hit@k          expected source page appears in retrieved chunks (answerable Qs)
  accuracy       LLM judge says answer matches expected answer (answerable Qs)
  citation_ok    a cited chunk contains an expected page (answerable Qs)
  refusal_acc    unanswerable Qs answered with "I don't know"
"""
import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from src import config, llm
from src.retrieve import retrieve

EVAL_DIR = Path(__file__).parent
QUESTIONS = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

JUDGE = """You grade answers to questions about university regulations.
Expected answer: {expected}
Model answer: {answer}
Does the model answer state the same key facts as the expected answer, without contradicting it?
Reply with exactly one word: CORRECT or INCORRECT."""


def load_questions() -> list[dict]:
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.startswith("//")]


def pages_hit(expected_pages: list[int], doc: str, items: list[dict]) -> bool:
    exp = set(expected_pages)
    return any(i["doc"] == doc and exp & set(i["pages"]) for i in items)


def judge(expected: str, answer: str) -> bool:
    out = llm.chat([{"role": "user", "content": JUDGE.format(expected=expected, answer=answer)}])
    return out["text"].strip().upper().startswith("CORRECT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="label for this run, e.g. hybrid-600")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    from src.answer import ask  # imported here so --retrieval-only needs no API key

    qs = load_questions()
    rows = []
    for q in qs:
        row = {"id": q["id"], "answerable": q["answerable"]}
        if args.retrieval_only:
            chunks = retrieve(q["question"])
            row["hit"] = pages_hit(q.get("source_pages", []), q.get("source_doc", ""), chunks) if q["answerable"] else None
        else:
            r = ask(q["question"])
            row.update(answer=r["answer"], refused=r["refused"], latency_s=r["latency_s"],
                       prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"])
            if q["answerable"]:
                row["hit"] = pages_hit(q["source_pages"], q["source_doc"], r["retrieved"])
                row["citation_ok"] = pages_hit(q["source_pages"], q["source_doc"], r["citations"])
                row["correct"] = (not r["refused"]) and judge(q["expected_answer"], r["answer"])
            else:
                row["correct"] = r["refused"]
        rows.append(row)
        print(f"{q['id']}: {'ok' if row.get('correct', row.get('hit')) else 'MISS'}")

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    pct = lambda xs: round(100 * sum(bool(x) for x in xs) / len(xs), 1) if xs else None
    summary = {
        "tag": args.tag,
        "time": datetime.now().isoformat(timespec="seconds"),
        "chunk_size": config.CHUNK_SIZE, "retrieval": config.RETRIEVAL_MODE,
        "top_k": config.TOP_K, "model": None if args.retrieval_only else config.LLM_MODEL,
        "n": len(rows), f"hit@{config.TOP_K}": pct([r["hit"] for r in ans]),
    }
    if not args.retrieval_only:
        lat = sorted(r["latency_s"] for r in rows)
        summary.update(
            accuracy=pct([r["correct"] for r in ans]),
            citation_ok=pct([r["citation_ok"] for r in ans]),
            refusal_acc=pct([r["correct"] for r in unans]),
            latency_p50=statistics.median(lat),
            latency_p95=lat[max(0, int(0.95 * len(lat)) - 1)],
            total_tokens=sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows),
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{args.tag}" if args.tag else stamp
    with open(RESULTS_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    with open(RESULTS_DIR / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
