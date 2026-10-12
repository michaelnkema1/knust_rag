# KNUST Regulations Assistant

A retrieval-augmented (RAG) assistant that answers KNUST students' questions about academic regulations, citing the document and page each answer comes from, and saying "I don't know" when the documents don't cover the question.

## How it works

```
question → retrieve top-k chunks (vector | BM25 | hybrid) → LLM with numbered sources → answer + [n] citations
```

- **Ingestion** (`src/ingest.py`): PDFs → cleaned text per page → overlapping word-window chunks that remember their pages → embedded with `bge-small-en-v1.5` into ChromaDB.
- **Retrieval** (`src/retrieve.py`): dense vector search, BM25 keyword search, or hybrid via reciprocal rank fusion.
- **Answering** (`src/answer.py`): strict prompt — sources only, cite every claim, refuse when unsupported.
- **LLM** (`src/llm.py`): any OpenAI-compatible endpoint (OpenAI, Groq, local Ollama), set in `.env`.
- **Evaluation** (`eval/run_eval.py`): hit@k, answer accuracy (LLM judge), citation correctness, refusal accuracy, latency and tokens.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                   # add your LLM key
```

Put the source PDFs in `data/raw/`, then:

```bash
python -m src.ingest                          # build the index
python -m src.answer "Where can I get counselling support?"
uvicorn app.main:app --reload                 # chat UI at http://localhost:8000
```

## Evaluation

```bash
python -m eval.run_eval --retrieval-only      # fast, free: retrieval hit@k only
python -m eval.run_eval --tag baseline        # full run with LLM + judge
CHUNK_SIZE=300 python -m src.ingest && python -m eval.run_eval --tag chunk300
RETRIEVAL_MODE=hybrid python -m eval.run_eval --tag hybrid
```

Each run appends a line to `eval/results/summary.jsonl`.

## Results

| Config | Hit@5 | Accuracy | Citations | Refusals | p95 latency |
| --- | --- | --- | --- | --- | --- |
| _fill in after week 3_ | | | | | |

## Limitations

- Answers are only as current as the PDFs indexed; some regulations may live in documents not yet included.
- Not an official KNUST service — confirm important decisions with your faculty office.
