# KNUST Regulations Assistant

A retrieval-augmented generation (RAG) assistant that answers KNUST students' questions about university regulations from the official **KNUST Students' Guide and Code of Conduct (2023)**. Every answer cites the PDF page it came from, and the assistant says "I don't know" when the guide doesn't cover the question.

The project is built around measurement: a reviewed 92-question eval set, automatic scoring, and experiments comparing retrieval methods, chunk sizes and models.

## How it works

```
question → retrieve top-k chunks (vector | BM25 | hybrid) → LLM with numbered sources → answer + [n] citations
```

| Part | File | What it does |
| --- | --- | --- |
| Ingestion | `src/ingest.py` | PDF → cleaned text per page → overlapping word-window chunks that remember their pages → embedded with `BAAI/bge-small-en-v1.5` into ChromaDB |
| Retrieval | `src/retrieve.py` | Dense vector search, BM25 keyword search, or hybrid (reciprocal rank fusion) |
| Answering | `src/answer.py` | Strict prompt: sources only, cite every claim, refuse when unsupported; parses `[2]`, `[1, 4]` and `[1][3]` citation styles |
| LLM client | `src/llm.py` | Any OpenAI-compatible endpoint, set in `.env`; 60 s timeout with automatic retries on rate limits and server errors |
| API + UI | `app/main.py`, `app/static/index.html` | FastAPI app and chat interface (see below) |
| Evaluation | `eval/` | Question drafting, review export, and a resumable scoring runner |

The default model is **Gemini 3.1 Flash-Lite** through Google's OpenAI-compatible endpoint. OpenAI, Groq or a local Ollama model work by changing three lines in `.env`.

## Chat interface

- **Clickable citations.** The `[n]` markers in an answer and the page tags under it open the source passage (the ~110 words of the retrieved chunk that best match the answer) with a link that opens the PDF at that page.
- **Feedback.** Each answer has thumbs up/down. A thumbs-down can carry a short comment. Feedback is appended to `data/feedback.jsonl` (git-ignored); thumbs-down rows are candidates for new eval questions.
- **Loading state.** Animated typing dots and a status line that changes the longer a request takes, so slow free-tier responses don't look frozen.
- **Theme.** Deep green `#0F372F` and gold `#ECBC6B`, with a dark mode that follows the system setting.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Chat page |
| `POST /ask` | `{"question": "..."}` → answer, citations (with excerpts), retrieved chunks, latency, tokens |
| `GET /docs-pdf/{name}` | Serves an indexed PDF from `data/raw/` (nothing else), e.g. `/docs-pdf/students_guide_2023.pdf#page=78` |
| `POST /feedback` | `{"question", "answer", "rating": "up" or "down", "comment"}` |

## Setup

Requires Python 3.10+.

```bash
./setup_env.sh                    # creates .venv; installs CPU-only PyTorch (~200 MB instead of ~2 GB) and requirements
cp .env.example .env              # then add your API key, base URL and model
source .venv/bin/activate
```

Example `.env` for Gemini:

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-3.1-flash-lite
```

`.env` is git-ignored, so your key never reaches GitHub.

Download the [2023 Students' Guide PDF](https://www.knust.edu.gh/sites/default/files/2023-06/Students%27%20Guide%20and%20Code%20of%20Conduct.pdf) into `data/raw/students_guide_2023.pdf`, then:

```bash
python -m src.ingest                                    # build the index (226 pages → 129 chunks at 600 words)
python -m src.answer "Where can I get counselling support?"
uvicorn app.main:app --reload                           # chat UI at http://localhost:8000
python -m pytest -q                                     # unit tests
```

## Evaluation

### The eval set

`eval/questions.jsonl` holds **92 questions: 79 answerable and 13 unanswerable**. They were built in three steps:

1. **Drafted** by `eval/generate_questions.py`: one student-style question per randomly sampled page, so each question's source page is known exactly, plus 15 questions the guide shouldn't answer.
2. **Reviewed by hand** in `eval/candidates.csv`: each row marked `keep`, `edit` or `drop`. 8 of 100 drafts were dropped.
3. **Exported** with `python -m eval.build_eval`, which only includes reviewed rows.

Page numbers are **PDF page numbers** (what a PDF viewer shows), not the numbers printed on the pages, which run about 19 lower.

### Running it

```bash
python -m eval.run_eval --retrieval-only --tag retr-vector     # free and fast: retrieval only, no LLM calls
python -m eval.run_eval --tag baseline-vector-600              # full run: answers + LLM judge
RETRIEVAL_MODE=hybrid python -m eval.run_eval --tag hybrid-600
CHUNK_SIZE=300 python -m src.ingest && python -m eval.run_eval --tag vector-300
```

Runs are **resumable**: every scored question is saved to `eval/results/<tag>.partial.jsonl` as it finishes. Re-running with the same `--tag` skips completed questions and retries any that errored, so a provider outage never loses work. Each finished run appends a summary line to `eval/results/summary.jsonl`.

### Metrics

| Metric | Meaning |
| --- | --- |
| Hit@5 | The question's source page is among the 5 retrieved chunks |
| Accuracy | An LLM judge rates the answer correct on the main point compared with the reference answer (secondary details may be omitted) |
| Citation correctness | A cited chunk contains the source page |
| Refusal accuracy | Unanswerable questions get "I don't know" |
| Latency | Seconds from question to answer (p50 / p95) |

## Results

### Retrieval only (79 answerable questions, top 5, 600-word chunks)

| Retrieval | Hit@5 |
| --- | --- |
| Vector | **75.9%** |
| Hybrid (vector + BM25) | 72.2% |
| BM25 (keyword) | 60.8% |

Plain vector search did best. The questions are paraphrased in a student's own words, so keyword matching struggles, and fusing it with vector search slightly lowers the result.

### End to end

92 questions (79 answerable, 13 unanswerable), top 5 chunks, 600-word chunks.

| Config | Hit@5 | Accuracy | Citations | Refusals | p50 / p95 latency |
| --- | --- | --- | --- | --- | --- |
| Baseline: vector, Gemini 3.1 Flash-Lite | 75.9% | 83.5% | 67.1% | 92.3% | 5.1 s / 21.1 s |

More rows (chunk size, retrieval method, second model) will be added as experiments finish.

### What the baseline shows

Breakdown of the 79 answerable questions:

| Outcome | Count |
| --- | --- |
| Source page retrieved, answer correct, citation correct | 52 |
| Source page **not** retrieved, but answer still correct | 12 |
| Source page retrieved, answer correct, but it cited a different chunk | 2 |
| Wrongly refused ("I don't know") although the answer was in the guide | 8 |
| Answered, but wrong | 5 |

- **The citation metric is stricter than it should be.** Each question is labelled with the single page it was drafted from, but the guide repeats many facts on other pages (for example, accommodation and security contacts). In 12 cases the assistant answered correctly from a different page, which counts as a citation miss (most of the gap between 83.5% accuracy and 67.1% citation correctness). Labelling questions with every page that answers them would make the metric fairer.
- **Over-refusal is the main answer error.** 8 of 79 answerable questions got "I don't know". In 4 of those the right page *was* retrieved, so the prompt is making the model too cautious. The other 4 are retrieval misses.
- **Refusals on unanswerable questions work.** 12 of 13 were correctly refused. The one exception asked about reporting bullying, which the guide's harassment section partly covers, so the question itself is borderline.
- **Latency is dominated by the free API tier.** The median is about 5 s, but busy periods push the slowest 5% past 20 s.

## Project structure

```
app/            FastAPI app and chat page
src/            config, ingestion, retrieval, LLM client, answering
eval/           question generation, review export, eval runner, questions.jsonl
tests/          unit tests
data/raw/       source PDFs (not committed)
setup_env.sh    one-time environment setup
```

## Limitations

- Answers are only as current as the indexed PDF. Some regulations, such as detailed exam rules, may live in documents not yet included.
- Most eval questions were drafted by an LLM from the guide, then reviewed by hand. Drafted questions can echo the guide's wording, which may make retrieval look better than it would on real student questions.
- Answer accuracy is scored by an LLM judge, which was checked on sample right and wrong answers but can still make mistakes.
- Latency figures come from a free API tier and vary with provider load.
- Not an official KNUST service. Confirm important decisions with your faculty office.
