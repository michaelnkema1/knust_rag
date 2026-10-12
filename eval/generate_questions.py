"""Draft eval questions from the source PDFs for human review.

  python -m eval.generate_questions            # writes eval/candidates.csv
  (review the CSV: set status to keep / edit / drop, fix wording)
  python -m eval.build_eval                    # writes eval/questions.jsonl from kept rows

Each answerable question is generated from ONE page, so its source page is known exactly.
"""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

from src import config, llm
from src.ingest import read_pages

OUT = Path(__file__).parent / "candidates.csv"
FIELDS = ["status", "id", "question", "expected_answer", "source_doc", "source_pages", "answerable", "notes"]

PAGE_PROMPT = """You are building a test set for a chatbot that answers KNUST students' questions using the official Students' Guide.

Below is one page of the guide (PDF page {page}).
Write ONE question a real KNUST student might ask that this page answers clearly.
- Phrase it the way a student would ask casually, in your own words. Do NOT copy phrases from the page.
- The answer must be fully supported by this page alone.
- If the page is a table of contents, an index, a list of names/phone numbers only, or has nothing a student would ask about, return {{"skip": true}}.

Return only JSON: {{"question": "...", "answer": "1-2 sentence answer using only the page"}}

PAGE TEXT:
{text}"""

UNANSWERABLE_PROMPT = """Write {n} realistic questions a KNUST student might ask a university-regulations chatbot that a general
KNUST Students' Guide and Code of Conduct would probably NOT answer (e.g. specific fees for a future year, other universities,
visa processes, specific lecturers, exam timetables, scholarships abroad, personal grades).
Vary the topics. Return only a JSON list of strings."""


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


def call(prompt: str, retries: int = 4):
    for attempt in range(retries):
        try:
            return parse_json(llm.chat([{"role": "user", "content": prompt}], temperature=0.7)["text"])
        except json.JSONDecodeError:
            continue  # model returned non-JSON; ask again
        except Exception as e:  # rate limits / transient errors on the free tier
            wait = 10 * (attempt + 1)
            print(f"  API error ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=85, help="answerable questions to draft")
    ap.add_argument("--unanswerable", type=int, default=15)
    ap.add_argument("--skip-front", type=int, default=20, help="skip cover/contents pages")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between calls (free-tier rate limits)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = []
    for pdf in sorted(config.RAW_DIR.glob("*.pdf")):
        pages = [p for p in read_pages(pdf) if p["page"] > args.skip_front and len(p["text"].split()) >= 120]
        random.Random(args.seed).shuffle(pages)
        print(f"{pdf.name}: {len(pages)} usable pages")

        for p in pages:
            if len(rows) >= args.n:
                break
            out = call(PAGE_PROMPT.format(page=p["page"], text=p["text"][:6000]))
            time.sleep(args.delay)
            if not out or out.get("skip") or not out.get("question"):
                print(f"  p.{p['page']}: skipped")
                continue
            rows.append({
                "status": "", "id": f"q{len(rows) + 1:03d}", "question": out["question"].strip(),
                "expected_answer": out.get("answer", "").strip(), "source_doc": p["doc"],
                "source_pages": str(p["page"]), "answerable": "yes", "notes": "",
            })
            print(f"  p.{p['page']}: {out['question'][:80]}")

    unans = call(UNANSWERABLE_PROMPT.format(n=args.unanswerable)) or []
    for q in unans[: args.unanswerable]:
        rows.append({
            "status": "", "id": f"q{len(rows) + 1:03d}", "question": q.strip(), "expected_answer": "",
            "source_doc": "", "source_pages": "", "answerable": "no",
            "notes": "CHECK the guide really does not answer this",
        })

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} candidates to {OUT}. Review them, then run: python -m eval.build_eval")


if __name__ == "__main__":
    main()
