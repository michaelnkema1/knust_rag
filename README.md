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
| API + UI | `app/main.py` | FastAPI `POST /ask` and a single-page chat interface |
| Evaluation | `eval/` | Question drafting, review export, and a resumable scoring runner |

The default model is **Gemini 3.1 Flash-Lite** through Google's OpenAI-compatible endpoint. OpenAI, Groq or a local Ollama model work by changing three lines in `.env`.

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

| Config | Hit@5 | Accuracy | Citations | Refusals | p95 latency |
| --- | --- | --- | --- | --- | --- |
| Baseline: vector, 600-word chunks, Gemini 3.1 Flash-Lite | | | | | |

_In progress._ More rows (chunk size, retrieval method, second model) will be added as experiments finish.

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
