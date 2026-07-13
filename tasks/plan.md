# Implementation Plan: FastAPI Docs RAG Q&A System

## Overview

Build a RAG pipeline (ingest → chunk → embed → index → retrieve → generate) over the FastAPI
documentation, exposed via a FastAPI HTTP API and packaged in Docker, with a RAGAS-based
evaluation harness measuring faithfulness, answer relevance, context precision, and context
recall. Built as vertical slices: get ingestion + retrieval + generation working as a plain
Python call first, then wrap it in an API, then containerize, then add the eval harness.

## Architecture Decisions

- **LangGraph over plain LangChain chains** — explicit state graph (retrieve node → generate
  node) is more inspectable/debuggable and better demonstrates systems thinking than an opaque
  chain.
- **Chroma embedded, no server container** — simplest deployment story for a portfolio project;
  persisted to `data/chroma`, rebuilt from `data/raw` if ever wiped.
- **Deterministic chunk IDs (hash of source path + content)** — makes re-ingestion idempotent
  without needing a separate "already indexed" tracking table.
- **Claude direct (`claude-haiku-4-5-20251001`) for generation + RAGAS judge, local
  `BAAI/bge-small-en-v1.5` embeddings** — no OpenAI credit available; Anthropic has no
  embeddings endpoint, so embeddings run locally at zero cost while the $17 Anthropic budget
  covers generation + eval LLM calls (changed from the original OpenAI-only plan — see SPEC.md
  Tech Stack).
- **Eval harness is informational, not CI-gated** — v1 goal is to produce and interpret the
  metrics, not build a gating pipeline; gating is a documented future step (see SPEC Boundaries).
- **Integration tests excluded from default `pytest` run** — they hit the real Anthropic API,
  which costs money; unit tests (mocked) are the fast/free default gate.

## Task List

### Phase 0: Foundation

- [ ] Task 1: Project scaffolding

### Checkpoint: Foundation
- [ ] `uv sync` installs cleanly
- [ ] `uv run ruff check .` and `uv run mypy src/` run clean on empty scaffold
- [ ] `.env.example` documents required keys

### Phase 1: Ingestion Pipeline

- [ ] Task 2: Fetch FastAPI docs corpus
- [ ] Task 3: Markdown-aware chunker
- [ ] Task 4: Embedding indexer + ingest CLI

### Checkpoint: Ingestion
- [ ] `uv run python scripts/ingest.py` populates `data/chroma` from a fresh clone
- [ ] Re-running ingest does not duplicate vectors (idempotency check)
- [ ] Unit tests for chunker pass

### Phase 2: Retrieval + QA Graph

- [ ] Task 5: Retriever wrapper
- [ ] Task 6: LangGraph QA graph

### Checkpoint: End-to-end Q&A (no API yet)
- [ ] A manual script/test call into the graph with a real FastAPI question returns a grounded
      answer with cited source chunks
- [ ] Unit tests (mocked LLM/embeddings) pass

### Phase 3: API

- [ ] Task 7: FastAPI app (`/health`, `/query`)

### Checkpoint: API
- [ ] `uv run uvicorn rag_project.api.main:app` serves; `curl localhost:8000/query` returns a
      grounded answer with citations in reasonable time

### Phase 4: Dockerization

- [ ] Task 8: Dockerfile + docker-compose

### Checkpoint: Docker
- [ ] `docker build` + `docker run --env-file .env` serves the same API with no extra manual
      steps

### Phase 5: Evaluation Harness

- [ ] Task 9: Curate eval set + dataset loader
- [ ] Task 10: RAGAS eval runner

### Checkpoint: Eval
- [ ] `uv run python -m rag_project.eval.run_eval` produces a report with all four RAGAS metrics
      over the full eval set

### Phase 6: Integration Tests + Polish

- [ ] Task 11: Integration tests (real API calls, marked, excluded from default run)
- [ ] Task 12: README + final pass against SPEC success criteria

### Checkpoint: Complete
- [ ] All SPEC.md success criteria verified
- [ ] `uv run pytest` (unit) and `uv run ruff check .` both pass clean
- [ ] Ready for human review

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Anthropic cost overrun on $17 budget | Med | Cheapest Claude model (Haiku 4.5) chosen for both generation and eval judge; embeddings run locally at zero cost; eval set capped at ~20-30 questions; integration tests excluded from routine runs |
| FastAPI docs corpus structure/size unknown until fetched | Low | Task 2 inspects the actual `docs/en/docs` tree before Task 3 finalizes chunking assumptions |
| Chroma re-ingestion creates duplicate vectors | Med | Deterministic IDs (hash of source path + content) used for upsert, verified by an idempotency check at the Ingestion checkpoint |
| RAGAS API/version churn breaks eval runner | Low | Pin RAGAS version in `pyproject.toml`; smoke-test on 2-3 questions before running the full eval set |
| LangGraph state/prompt bugs only surface with real API calls (costs money to debug) | Med | Unit tests mock the LLM/embeddings for logic correctness; real calls reserved for the checkpoint manual checks and integration tests |

## Open Questions

- Exact eval set composition (~20-30 Q&A pairs) is curated during Task 9, not blocking earlier
  phases.
- Minimal frontend UI remains out of scope for this plan (deferred per SPEC).
