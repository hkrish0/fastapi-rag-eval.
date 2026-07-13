# Spec: FastAPI Docs RAG Q&A System

## Objective

A document Q&A system that answers questions about the FastAPI documentation, grounded in
retrieval over the actual docs — plus an evaluation harness that measures whether those
answers are actually correct (faithful to source, relevant, well-retrieved), not just fluent.

**Why:** This is a portfolio project supporting a career transition into AI/ML and freelance
work on Upwork. The differentiator is the evaluation harness (RAGAS) — most RAG demos skip
measuring correctness entirely. Demonstrating rigor here signals systems-engineering judgment
to technical reviewers and gives non-technical clients (Upwork) a concrete, demoable metric
("87% faithfulness, 92% answer relevance") rather than just a chat demo.

**User:** A technical reviewer (hiring manager, freelance client) evaluating whether the author
can build and rigorously validate a production-shaped RAG system.

**Success looks like:** Clone the repo, run one ingestion command and one `docker run`, ask a
FastAPI question via the API, get a grounded answer with cited sources, and run the eval suite
to see quantified retrieval/answer-quality metrics.

## Tech Stack

- Python 3.11+, dependency management via `uv`
- LangGraph for the retrieval → generation pipeline (explicit state graph, not an opaque chain)
- Chroma (embedded/persisted to local disk, no separate DB server/container)
- FastAPI for the HTTP API
- Generation + RAGAS eval judge: Claude direct via `langchain-anthropic`, model
  `claude-haiku-4-5-20251001` — chosen for cost (~$17 Anthropic API budget, no OpenAI credit
  available)
- Embeddings: local `BAAI/bge-small-en-v1.5` via `sentence-transformers`/`langchain-huggingface`
  — Anthropic has no embeddings endpoint, so embeddings run locally at zero API cost (corpus is
  ~2100 chunks, fast enough on CPU)
- RAGAS for evaluation (faithfulness, answer relevance, context precision, context recall),
  using the local embeddings above and Claude Haiku as the judge LLM
- Docker for packaging/deployment
- ruff (lint + format), mypy (type checking), pytest (tests)

## Commands

```
Install deps:      uv sync
Run dev server:    uv run uvicorn rag_project.api.main:app --reload --port 8000
Ingest docs:       uv run python scripts/ingest.py
Run eval harness:  uv run python -m rag_project.eval.run_eval
Run all tests:     uv run pytest
Run unit only:     uv run pytest tests/unit
Run single test:   uv run pytest tests/unit/test_chunker.py::test_chunk_respects_max_tokens -v
Lint:              uv run ruff check --fix .
Format:            uv run ruff format .
Type check:        uv run mypy src/
Docker build:      docker build -t rag-project .
Docker run:        docker run -p 8000:8000 --env-file .env rag-project
```

## Project Structure

```
src/rag_project/
  config.py              → pydantic-settings: API keys, model names, paths, chunk size
  ingestion/
    fetch_docs.py         → pulls FastAPI docs markdown (docs/en/docs from tiangolo/fastapi)
    chunker.py            → markdown-aware chunking
    indexer.py            → embeds chunks, upserts into Chroma (idempotent)
  retrieval/
    retriever.py          → Chroma similarity-search wrapper
  graph/
    qa_graph.py           → LangGraph: retrieve node → generate node → response w/ citations
  api/
    main.py                → FastAPI app, routes (/query, /health)
    schemas.py              → pydantic request/response models
  eval/
    dataset.py             → loads eval_set.jsonl (question, ground_truth, contexts)
    run_eval.py            → runs RAGAS metrics against the QA graph, writes report
    eval_set.jsonl          → ~20-30 hand-curated FastAPI-docs Q&A pairs

data/
  raw/                     → cloned FastAPI docs markdown (gitignored, regenerable)
  chroma/                  → persisted Chroma DB (gitignored, regenerable via ingest)

reports/eval/               → timestamped RAGAS score reports (committed as dated samples)

scripts/ingest.py            → CLI entrypoint that runs fetch → chunk → index

tests/
  unit/                      → chunking, schema, retrieval-logic tests (embeddings/LLM mocked)
  integration/                → real Chroma + real Anthropic API calls, marked @pytest.mark.integration,
                                 excluded from default `pytest` run (costs money, run manually)

Dockerfile, docker-compose.yml, pyproject.toml, .env.example, README.md, SPEC.md
```

## Code Style

- Type hints on all function signatures; `mypy` must pass.
- Pydantic models at every I/O boundary (API requests/responses, eval dataset rows, config).
- Prefer plain functions for LangGraph nodes; use a class only when node logic needs shared
  state beyond what fits in the LangGraph state object.
- No inline comments explaining *what* code does — only *why*, for non-obvious constraints
  (e.g. why a chunk size was picked, why a retry exists).

```python
class RetrievedChunk(BaseModel):
    content: str
    source_path: str
    score: float

def retrieve(state: QAState, retriever: Retriever, k: int = 4) -> QAState:
    chunks = retriever.similarity_search(state.question, k=k)
    return state.model_copy(update={"chunks": chunks})
```

## Testing Strategy

- **Unit tests** (`tests/unit`, default `pytest` run): chunking correctness, schema validation,
  retriever logic — all with mocked embeddings/LLM calls. Must pass before every commit.
- **Integration tests** (`tests/integration`, `@pytest.mark.integration`): real Chroma + real
  Anthropic API calls end-to-end. Excluded from default `pytest` run (costs money); run manually
  before merging changes to ingestion, retrieval, or the QA graph.
- **Eval harness** (RAGAS, `run_eval.py`): not a pass/fail test suite. Generates a scored report
  (faithfulness, answer relevance, context precision, context recall) over the hand-curated
  eval set. Informational only for v1 — no CI gate on scores. Re-run and compare reports whenever
  prompt, chunking, or retrieval logic changes.

## Boundaries

- **Always:** run `ruff check` and unit `pytest` before committing; keep secrets in `.env`
  (gitignored, never committed); regenerate an eval report after any change to prompts,
  chunking, or retrieval logic and note the score delta.
- **Ask first:** adding new paid API dependencies or services, changing the vector store or
  embedding/generation provider, changing the chunking strategy (requires full re-ingestion),
  moving from informational eval to CI-gated thresholds.
- **Never:** commit `.env` or API keys; commit `data/raw` or `data/chroma` (regenerable via
  ingestion); call real Anthropic APIs from unit tests (must be mocked); remove or skip a failing
  test without explicit approval.

## Success Criteria

- `docker build` + `docker run --env-file .env` serves a working API with no manual steps beyond
  supplying API keys.
- `uv run python scripts/ingest.py` chunks and indexes the FastAPI docs corpus into Chroma,
  and is idempotent (re-running does not duplicate vectors).
- `POST /query` with a FastAPI-docs question returns a grounded answer plus cited source chunks
  in a reasonable time (target: under ~5s per query).
- `uv run python -m rag_project.eval.run_eval` produces a report with faithfulness, answer
  relevance, context precision, and context recall scores over the ~20-30 question eval set.
- `uv run pytest` (unit tests) and `uv run ruff check` both pass clean.
- README documents setup, ingestion, running the API, and running eval clearly enough for a
  freelance client or portfolio reviewer to run the project themselves.

## Open Questions

- Exact eval set size/composition (~20-30 Q&A pairs) will be curated during implementation —
  not blocking the plan.
- Whether to eventually add a minimal frontend (deferred out of v1 scope; API-only for now).
