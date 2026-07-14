# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A RAG Q&A system over the FastAPI documentation, with a RAGAS evaluation harness measuring
answer faithfulness/relevance and retrieval precision/recall. Full requirements, tech stack
rationale, and boundaries live in `SPEC.md` — read it before making architectural decisions.
The implementation plan and per-task acceptance criteria live in `tasks/plan.md` and
`tasks/todo.md`; check `tasks/todo.md` for what's actually built vs. still planned before
assuming a component exists.

## Commands

```
Install deps:      uv sync
Ingest docs:       uv run python scripts/ingest.py
Run all tests:     uv run pytest
Run single test:   uv run pytest tests/unit/test_chunker.py::test_short_doc_produces_a_single_chunk -v
Lint:              uv run ruff check --fix .
Format:            uv run ruff format .
Type check:        uv run mypy src/ scripts/
```

`uv run uvicorn rag_project.api.main:app` and `uv run python -m rag_project.eval.run_eval` are
implemented (Tasks 7, 10). Docker build/run themselves are not yet verified in this environment
(no Docker installed) — see `tasks/todo.md` Task 8. Check `tasks/todo.md` for what's checked off
before assuming a component exists.

`uv run pytest` only runs `tests/unit` by default — `pyproject.toml`'s `addopts` excludes the
`integration` marker because integration tests hit the real Anthropic API and cost money. Run
`uv run pytest -m integration` explicitly when those exist and need verifying.

## Architecture

Pipeline: `fetch_docs` → `chunker` → `indexer` (Phase 1, done) → `retriever` → LangGraph
`qa_graph` (Phase 2) → FastAPI `/query` (Phase 3) → Docker (Phase 4) → RAGAS `run_eval` (Phase 5).

**Provider split (non-obvious, differs from most RAG tutorials):** generation and the RAGAS
judge LLM both use Claude (`claude-haiku-4-5-20251001` via `langchain-anthropic`), but
embeddings run locally via `sentence-transformers` (`BAAI/bge-small-en-v1.5` via
`langchain-huggingface`) — Anthropic has no embeddings endpoint, and there's no OpenAI credit
available for this project. Any new component that needs embeddings or generation should reuse
`config.py`'s `embedding_model`/`generation_model` settings rather than hardcoding a provider.

**Ingestion (`src/rag_project/ingestion/`):**
- `fetch_docs.py` downloads the `docs/en/docs` subtree from `tiangolo/fastapi`'s GitHub tarball
  (not a git clone — avoids needing git in the Docker image). Tar member paths are resolved and
  checked against `dest_dir` before writing (tar-slip guard) since tar contents are untrusted
  external data.
- `chunker.py` splits markdown into packing units per *line*, not per blank-line paragraph — a
  blank-line-based splitter was the original approach but collapsed FastAPI's
  `release-notes.md` (a long bullet list with no blank lines between items) into one 25,000-char
  chunk. Fenced code blocks are the one exception: kept atomic even if they alone exceed
  `chunk_size`, since splitting a code block mid-way would break retrieval quality worse than an
  oversized chunk would.
- `indexer.py` upserts into Chroma using a deterministic chunk ID (`sha256(source_path + content)`),
  which is what makes re-running `scripts/ingest.py` idempotent — `Chroma.add_texts` upserts on
  an existing ID rather than erroring or duplicating. `get_vector_store()` rebuilds the embedding
  model + Chroma client on every call; that's fine for the one-shot CLI but should be cached
  (e.g. `lru_cache`) once a retriever or API endpoint calls it per-request, or every query will
  reload the embedding model from disk.

**Config (`config.py`):** pydantic-settings, reads `.env` (never committed — see `.env.example`
for the required keys). `ANTHROPIC_API_KEY` is required; everything else has a default.

## Conventions

- Type hints required everywhere; `mypy --strict` must pass.
- Pydantic models at every I/O boundary (chunks, API schemas, eval rows, config) — see `Chunk`
  in `chunker.py` for the pattern.
- No inline comments explaining *what* code does — only non-obvious *why* (see the chunker
  module for examples of the bar to clear).
- `data/raw/` and `data/chroma/` are gitignored and regenerable via `scripts/ingest.py` — never
  commit them, and don't treat their absence as a bug.
