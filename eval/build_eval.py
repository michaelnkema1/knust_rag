"""Turn reviewed eval/candidates.csv into eval/questions.jsonl.

Status column: keep | edit (you fixed the row, keep it) | drop | blank (= not reviewed yet).
By default only keep/edit rows are exported; --include-unreviewed also exports blanks.
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-unreviewed", action="store_true")
    args = ap.parse_args()

    with open(HERE / "candidates.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok = {"keep", "edit"} | ({""} if args.include_unreviewed else set())
    kept = [r for r in rows if r["status"].strip().lower() in ok]
    unreviewed = sum(1 for r in rows if not r["status"].strip())

    with open(HERE / "questions.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(kept, start=1):
            pages = [int(x) for x in r["source_pages"].replace(";", ",").split(",") if x.strip()]
            f.write(json.dumps({
                "id": f"q{i:03d}",
                "question": r["question"].strip(),
                "expected_answer": r["expected_answer"].strip(),
                "source_doc": r["source_doc"].strip(),
                "source_pages": pages,
                "answerable": r["answerable"].strip().lower() in {"yes", "true", "1"},
            }, ensure_ascii=False) + "\n")

    n_ans = sum(r["answerable"].strip().lower() in {"yes", "true", "1"} for r in kept)
    print(f"Exported {len(kept)} questions ({n_ans} answerable, {len(kept) - n_ans} unanswerable).")
    if unreviewed and not args.include_unreviewed:
        print(f"{unreviewed} rows are still unreviewed (blank status) and were left out.")


if __name__ == "__main__":
    main()
