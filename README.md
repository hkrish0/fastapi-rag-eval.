# FastAPI Docs RAG Q&A System

A RAG Q&A system over the [FastAPI documentation](https://fastapi.tiangolo.com/), with a
[RAGAS](https://docs.ragas.io/) evaluation harness that measures whether answers are actually
correct — faithful to source, relevant, well-retrieved — not just fluent.

Ask a question about FastAPI, get a grounded answer with cited source chunks, and run the eval
suite to see quantified retrieval/answer-quality metrics.

## Architecture

```
fetch_docs → chunker → indexer  (ingestion, offline)
                            ↓
                    Chroma (local, persisted)
                            ↓
retriever → LangGraph qa_graph (retrieve → generate) → FastAPI /query
                            ↑
              RAGAS run_eval scores this pipeline against a hand-curated eval set
```

Generation and the RAGAS judge LLM both use **Claude** (`claude-haiku-4-5-20251001` via
`langchain-anthropic`). Embeddings run **locally** via `sentence-transformers`
(`BAAI/bge-small-en-v1.5`) — Anthropic has no embeddings endpoint, and this project runs on a
small Anthropic-only API budget with no OpenAI credit.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY` to a real key. The other settings (embedding model,
generation model, chunk size, data paths) have working defaults — see `.env.example`.

## Running ingestion

Downloads the FastAPI docs corpus, chunks it, embeds it locally, and indexes it into a
persisted Chroma collection at `data/chroma`:

```bash
uv run python scripts/ingest.py
```

Safe to re-run — chunk IDs are deterministic (hash of source path + content), so re-ingesting
upserts in place instead of duplicating vectors. A fresh run indexes ~2100 chunks from ~154
fetched markdown files.

## Running the API

```bash
uv run uvicorn rag_project.api.main:app --reload --port 8000
```

```bash
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I define a path parameter?"}'
```

Returns `{"answer": "...", "sources": [{"content": ..., "source_path": ..., "score": ...}, ...]}`.
`GET /health` returns `{"status": "ok"}`.

The QA graph (embedding model + Chroma client) loads once on the first request and is cached
for the life of the process — the first request after startup takes ~10-15s, subsequent
requests are typically 3-5s.

## Running with Docker

```bash
docker build -t rag-project .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data rag-project
```

The container doesn't run ingestion itself — run `scripts/ingest.py` on the host first (or
mount a `data/` directory that already has a populated `data/chroma`) so `/query` has a corpus
to retrieve against.

> **Note:** the Docker build/run steps above have not been verified end-to-end in every
> environment this project has been developed in (Docker wasn't available in one of them). The
> Dockerfile's dependency-install and runtime sequence was verified by replicating it in an
> isolated venv outside Docker. If you hit an issue running the container, please file it.

## Running the eval harness

Scores the QA graph against a 25-question hand-curated eval set
(`src/rag_project/eval/eval_set.jsonl`, spanning 18 FastAPI doc topics) on RAGAS's faithfulness,
answer relevance, context precision, and context recall metrics, using Claude as the judge LLM
and the same local embeddings as retrieval.

```bash
# Cheap smoke test first (a few real Anthropic API calls)
uv run python -m rag_project.eval.run_eval --limit 3

# Full eval set (~100 real Anthropic API calls, a few minutes)
uv run python -m rag_project.eval.run_eval
```

Writes a timestamped JSON report to `reports/eval/eval_report_<timestamp>.json`, containing
per-question and averaged scores. Sample report committed at
`reports/eval/eval_report_20260714T191400Z.json`
(faithfulness 0.888, answer_relevancy 0.894, context_precision 0.681, context_recall 0.693).

The eval harness is informational, not a pass/fail gate — re-run it and compare reports whenever
prompt, chunking, or retrieval logic changes.

## Running tests

```bash
uv run pytest              # unit tests only (mocked embeddings/LLM calls, no API cost)
uv run pytest -m integration   # real Chroma + real Anthropic API calls, costs money, run manually
```

## Linting and type checking

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy src/ scripts/
```

## Project layout

See `SPEC.md` for the full requirements and architecture rationale, and `tasks/todo.md` for
what's built vs. planned.
