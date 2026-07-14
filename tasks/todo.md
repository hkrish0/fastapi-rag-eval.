# Task List: FastAPI Docs RAG Q&A System

## Phase 0: Foundation

### Task 1: Project scaffolding

**Description:** Set up the uv-managed Python project skeleton: `pyproject.toml` with all
planned dependencies (langgraph, langchain-core, chromadb, langchain-chroma, langchain-anthropic,
sentence-transformers, langchain-huggingface, fastapi, uvicorn, pydantic-settings, ragas, ruff,
mypy, pytest, pytest-mock), directory structure per SPEC.md, `config.py` (pydantic-settings
reading API keys/model names/paths from env), `.env.example`, `.gitignore` (excludes `data/raw`,
`data/chroma`, `.env`).

*(Updated after Task 1: switched from OpenAI to Claude (generation/eval judge) + local
sentence-transformers (embeddings) — no OpenAI credit available. See SPEC.md Tech Stack.)*

**Acceptance criteria:**
- [x] `uv sync` installs all dependencies without errors
- [x] `src/rag_project/` package structure exists matching SPEC.md's Project Structure section
- [x] `config.py` loads settings from `.env` via pydantic-settings with sane defaults/required
      fields for `ANTHROPIC_API_KEY`, embedding model name, generation model name, chunk size,
      Chroma persist path

**Verification:**
- [x] `uv sync` exits 0
- [x] `uv run ruff check .` passes clean
- [x] `uv run mypy src/` passes clean
- [x] Manual check: settings-loading mechanism verified with a dummy key (user to add real
      `.env` from `.env.example` before Task 4, which needs to download the local embedding
      model; Task 6+ needs a real `ANTHROPIC_API_KEY`)

**Dependencies:** None

**Files likely touched:**
- `pyproject.toml`, `.env.example`, `.gitignore`
- `src/rag_project/__init__.py`, `src/rag_project/config.py`

**Estimated scope:** Small (1-2 files of real logic, rest is scaffolding)

---

## Phase 1: Ingestion Pipeline

### Task 2: Fetch FastAPI docs corpus

**Description:** Implement `fetch_docs.py` to pull the `docs/en/docs` markdown tree from the
`tiangolo/fastapi` GitHub repo into `data/raw/`, exposed as a callable function (not just a
shell script) so `scripts/ingest.py` can call it directly.

**Acceptance criteria:**
- [x] Running fetch populates `data/raw/` with the FastAPI markdown docs tree
- [x] Re-running fetch is safe (overwrites/updates rather than erroring or duplicating)
- [x] Function signature allows specifying a target ref/branch (default: latest default branch,
      `master`)

**Verification:**
- [x] Manual check: fetched real corpus, 154 markdown files landed in `data/raw/`
      (e.g. `data/raw/tutorial/first-steps.md`)
- [x] Unit test with a mocked `httpx.get` call verifies extraction filters to the
      `docs/en/docs/` subpath and rerun doesn't duplicate/leave stale files

**Dependencies:** Task 1

**Files likely touched:**
- `src/rag_project/ingestion/fetch_docs.py`
- `tests/unit/test_fetch_docs.py`

**Estimated scope:** Small (1 file + 1 test file)

---

### Task 3: Markdown-aware chunker

**Description:** Implement `chunker.py` that splits fetched markdown files into retrieval-sized
chunks, respecting markdown structure (don't split mid-code-block, prefer splitting on headers)
and attaching source metadata (file path, section heading) to each chunk.

**Acceptance criteria:**
- [x] Chunks stay under the configured max size (from `config.py`), except a fenced code block
      that alone exceeds max size (kept whole rather than split — verified against the real
      corpus, largest such case is 3600 chars of terminal output in `deployment/server-workers.md`)
- [x] Code blocks are never split across chunk boundaries
- [x] Each chunk carries its source file path and nearest heading as metadata

**Verification:**
- [x] `uv run pytest tests/unit/test_chunker.py -v` passes (5 tests)
- [x] Tests cover: nested headers, a large code block, a doc shorter than max chunk size, and a
      regression case (a long blank-line-free bullet list, found via manual verification against
      `release-notes.md`, which the initial blank-line-based splitter treated as one 24,959-char
      atomic block — fixed by splitting non-code content per line instead of per paragraph)
- [x] Manual check: ran chunker against the full real corpus (154 files → 2102 chunks)

**Dependencies:** Task 1

**Files likely touched:**
- `src/rag_project/ingestion/chunker.py`
- `tests/unit/test_chunker.py`

**Estimated scope:** Small-Medium (1 file, several edge-case tests)

---

### Task 4: Embedding indexer + ingest CLI

**Description:** Implement `indexer.py`, which embeds chunks locally via
`sentence-transformers` (`BAAI/bge-small-en-v1.5`, no API cost) and upserts them into a
persisted Chroma collection using deterministic IDs (hash of source path + chunk content) so
re-running is idempotent. Wire `fetch_docs.py` → `chunker.py` → `indexer.py` together in
`scripts/ingest.py` as a single CLI entrypoint.

**Acceptance criteria:**
- [x] `uv run python scripts/ingest.py` runs fetch → chunk → embed → index end to end
- [x] Chroma collection at `data/chroma` contains the expected chunk count after a fresh run
      (2102 chunks from 154 fetched files)
- [x] Running ingest a second time does not duplicate vectors (same collection count)

**Verification:**
- [x] Manual check: ran ingest twice — 2102 chunks before, 2102 after (idempotent via
      `Chroma.add_texts`'s upsert-on-existing-id behavior + our deterministic hash IDs)
- [x] Unit test mocks the vector store and verifies deterministic ID generation
      (same input → same ID, different content/source → different ID)

**Dependencies:** Tasks 2, 3

**Files likely touched:**
- `src/rag_project/ingestion/indexer.py`
- `scripts/ingest.py`
- `tests/unit/test_indexer.py`

**Estimated scope:** Medium (2 files + test, first real API cost incurred here — small, since
embeddings are cheap)

---

## Checkpoint: Foundation + Ingestion
- [x] `uv run python scripts/ingest.py` succeeds against the real FastAPI docs corpus
- [x] Idempotency verified (re-run doesn't duplicate)
- [x] `uv run pytest tests/unit` passes, `uv run ruff check .` clean
- [ ] **Review with human before proceeding to Phase 2**

---

## Phase 2: Retrieval + QA Graph

### Task 5: Retriever wrapper

**Description:** Implement `retriever.py`, a thin wrapper over the Chroma collection's
similarity search, returning `RetrievedChunk` pydantic models (content, source_path, score) and
accepting a configurable `k`.

**Acceptance criteria:**
- [x] Given a query string, returns top-`k` `RetrievedChunk` results ordered by relevance
- [x] Empty/no-match case returns an empty list, not an error

**Verification:**
- [x] `uv run pytest tests/unit/test_retriever.py -v` passes (Chroma calls mocked)
- [x] Manual check: query the real indexed corpus for "How do I define a path parameter?" and
      confirm the top result is a relevant FastAPI docs chunk (top hit: `tutorial/path-params.md`,
      score 0.348, ascending L2 distance from Chroma's `similarity_search_with_score`)

**Dependencies:** Task 4 (needs a populated Chroma collection to test against manually)

**Files likely touched:**
- `src/rag_project/retrieval/retriever.py`
- `tests/unit/test_retriever.py`

**Estimated scope:** Small

---

### Task 6: LangGraph QA graph

**Description:** Implement `qa_graph.py`: a LangGraph state graph with a `retrieve` node (calls
the retriever) and a `generate` node (calls `gpt-4o-mini` with retrieved chunks as context,
producing an answer plus the list of source chunks it cited). Define the `QAState` pydantic
model threading through the graph.

**Acceptance criteria:**
- [x] Invoking the graph with a question returns an answer string and a list of cited source
      chunks (path + snippet)
- [x] The generate node's prompt instructs the model to answer only from provided context and
      to say so explicitly if the context doesn't contain the answer

**Verification:**
- [x] `uv run pytest tests/unit/test_qa_graph.py -v` passes (LLM call mocked, verifies graph
      wiring/state passing)
- [x] Manual check: invoked the graph directly with "How do I define a path parameter in
      FastAPI?" — grounded, cited answer (top source `tutorial/path-params.md`); also confirmed
      an out-of-scope question ("boiling point of tungsten") gets an explicit "context doesn't
      contain the answer" refusal instead of a hallucinated answer

**Dependencies:** Task 5

**Files likely touched:**
- `src/rag_project/graph/qa_graph.py`
- `tests/unit/test_qa_graph.py`

**Estimated scope:** Medium

---

## Checkpoint: End-to-end Q&A (no API yet)
- [x] Manual invocation of the QA graph with a real question returns a grounded, cited answer
- [x] `uv run pytest tests/unit` passes
- [ ] **Review with human before proceeding to Phase 3**

---

## Phase 3: API

### Task 7: FastAPI app

**Description:** Implement `api/schemas.py` (QueryRequest/QueryResponse pydantic models) and
`api/main.py` (FastAPI app with `GET /health` and `POST /query`, the latter invoking the QA
graph and returning the answer + citations).

**Acceptance criteria:**
- [x] `GET /health` returns 200
- [x] `POST /query` with `{"question": "..."}` returns `{"answer": "...", "sources": [...]}`
- [x] Invalid/empty question body returns a 422 via pydantic validation (no custom error
      handling needed beyond FastAPI's default)

**Verification:**
- [x] `uv run pytest tests/unit/test_api.py -v` passes (uses FastAPI `TestClient`, QA graph
      mocked)
- [x] Manual check: ran `uv run uvicorn rag_project.api.main:app`, then
      `curl -X POST localhost:8123/query -H 'Content-Type: application/json' -d '{"question": "How do I define a path parameter?"}'`
      returned a grounded answer citing `tutorial/path-params.md`; also confirmed an empty
      question returns 422 with a pydantic `string_too_short` detail

**Dependencies:** Task 6

**Files likely touched:**
- `src/rag_project/api/main.py`
- `src/rag_project/api/schemas.py`
- `tests/unit/test_api.py`

**Estimated scope:** Small

---

## Checkpoint: API
- [x] Manual curl check above succeeds with real retrieval + generation
- [x] `uv run pytest tests/unit` passes
- [ ] **Review with human before proceeding to Phase 4**

---

## Phase 4: Dockerization

### Task 8: Dockerfile + docker-compose

**Description:** Write a `Dockerfile` (multi-stage or single-stage with `uv`) that installs
deps and runs the uvicorn server, plus a `docker-compose.yml` for local convenience (mounts
`data/` as a volume so Chroma persists across container restarts, reads `.env`).

**Acceptance criteria:**
- [ ] `docker build -t rag-project .` succeeds
- [ ] `docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data rag-project` serves the
      API and `/query` works against the already-ingested Chroma data
- [ ] No manual steps inside the container required beyond providing `.env` and mounted data

**Verification:**
- [ ] Manual check: build + run, then repeat the `curl /query` check from Task 7 against the
      containerized API
      **Not run — Docker isn't installed in this environment.** Substitute check performed
      instead: replicated the Dockerfile's exact `uv sync --frozen --no-install-project --no-dev`
      then `uv sync --frozen --no-dev` layer sequence in an isolated directory (fresh venv, no
      access to the project's existing `.venv`), then ran that venv's `uvicorn` binary against
      the real `.env` + `data/chroma` — `/health` returned 200 and `/query` returned a grounded
      answer. This confirms the dependency-install and runtime path the Dockerfile encodes, but
      **the actual `docker build`/`docker run` commands still need to be run by a human** before
      checking the acceptance criteria above.

**Dependencies:** Task 7

**Files likely touched:**
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`

**Estimated scope:** Small

---

## Checkpoint: Docker
- [ ] Containerized API answers a real query correctly
- [ ] **Review with human before proceeding to Phase 5**

---

## Phase 5: Evaluation Harness

### Task 9: Curate eval set + dataset loader

**Description:** Hand-curate ~20-30 question/ground-truth-answer pairs covering a spread of the
FastAPI docs (path params, dependency injection, request bodies, middleware, etc.) into
`eval_set.jsonl`. Implement `dataset.py` to load and validate these rows as pydantic models.

**Acceptance criteria:**
- [x] `eval_set.jsonl` has 20-30 rows, each with `question` and `ground_truth` fields, spanning
      at least 5 distinct FastAPI doc topics (25 rows spanning 18 topics: path params, query
      params + validation, request body, dependencies (+ yield), middleware, CORS, error
      handling, background tasks, response model, response status code, security, testing,
      static files, cookie/header params, request files)
- [x] `dataset.py` loads the file into a list of validated pydantic rows, raising a clear error
      on a malformed row

**Verification:**
- [x] `uv run pytest tests/unit/test_dataset.py -v` passes (includes a malformed-row rejection
      test, 5 tests total)
- [x] Manual check: every row was written directly from the fetched `data/raw` corpus text
      (not from memory) while authoring the file; spot-checked 5 rows (query-param optionality,
      dependencies-with-yield, CORS middleware, `response_model_exclude_unset`, header
      underscore-conversion) against the source markdown — all accurate

**Dependencies:** Task 4 (needs the real corpus to write accurate ground-truth answers against)

**Files likely touched:**
- `src/rag_project/eval/eval_set.jsonl`
- `src/rag_project/eval/dataset.py`
- `tests/unit/test_dataset.py`

**Estimated scope:** Small

---

### Task 10: RAGAS eval runner

**Description:** Implement `run_eval.py`: for each row in the eval set, invoke the QA graph to
get an answer + retrieved contexts, then run RAGAS's faithfulness, answer relevance, context
precision, and context recall metrics, writing a timestamped report (JSON or Markdown) to
`reports/eval/`.

**Acceptance criteria:**
- [x] Running the eval produces a report file containing all four metric scores, both
      per-question and averaged
- [x] Report filename includes a timestamp so successive runs don't overwrite each other

**Verification:**
- [x] Manual check: `uv run python -m rag_project.eval.run_eval --limit 3` smoke test confirmed
      report structure, then ran the full 25-question eval set (real Claude generation + judge
      calls, ~2m45s). Averages: faithfulness 0.888, answer_relevancy 0.894, context_precision
      0.681, context_recall 0.693. Report committed at
      `reports/eval/eval_report_20260714T191400Z.json`. Note: ragas 0.4.3 as originally locked
      was unimportable (`ModuleNotFoundError: langchain_community.chat_models.vertexai`, an
      upstream ragas bug against modern langchain-community) — fixed by adding
      `langchain-community<0.4` as an explicit dependency, which pins to the last release still
      shipping that legacy submodule without downgrading langchain-core/langgraph/langchain-anthropic
- [x] `uv run pytest tests/unit/test_run_eval.py -v` passes (RAGAS/LLM calls mocked, verifies
      report-writing logic and averaging, 6 tests)

**Dependencies:** Tasks 6, 9

**Files likely touched:**
- `src/rag_project/eval/run_eval.py`
- `tests/unit/test_run_eval.py`

**Estimated scope:** Medium

---

## Checkpoint: Eval
- [x] Full eval set run produces a report with all four RAGAS metrics
- [ ] **Review with human before proceeding to Phase 6**

---

## Phase 6: Integration Tests + Polish

### Task 11: Integration tests

**Description:** Write `tests/integration/` tests covering the real end-to-end path (ingest a
small fixture corpus or reuse the real one → retrieve → generate) using the real Anthropic API + Chroma
calls, marked `@pytest.mark.integration` and excluded from the default `pytest` collection via
`pyproject.toml`/`pytest.ini` config.

**Acceptance criteria:**
- [x] `uv run pytest` (default) does not run these tests
- [x] `uv run pytest -m integration` runs them and they pass against the real corpus

**Verification:**
- [x] Both commands above produce the expected inclusion/exclusion behavior: default run shows
      "31 passed, 4 deselected"; `-m integration` shows "4 passed, 31 deselected" (real Chroma +
      real Anthropic API, ~40s). Covers: retriever returns a relevant chunk for a known query,
      the QA graph gives a grounded cited answer for a real question, the QA graph declines an
      out-of-scope question instead of hallucinating, and the full `/query` endpoint works
      end-to-end with no mocks. pytest marker config (`addopts = "-m 'not integration'"`) was
      already in place from Task 1, so only the test file itself was new.

**Dependencies:** Tasks 6, 7

**Files likely touched:**
- `tests/integration/test_end_to_end.py`
- `pyproject.toml` (pytest marker config)

**Estimated scope:** Small

---

### Task 12: README + final SPEC review

**Description:** Write `README.md` covering setup (`uv sync`, `.env`), running ingestion,
running the dev server, running Docker, and running the eval harness. Do a final pass checking
every SPEC.md Success Criteria item is actually met.

**Acceptance criteria:**
- [ ] A reader unfamiliar with the project can follow the README to get a working local
      instance and a working eval report
- [ ] Every bullet in SPEC.md's Success Criteria section is checked off with evidence (command
      run + observed result)

**Verification:**
- [ ] Manual check: follow the README from a clean clone (or mentally trace each command)
- [ ] Re-run `uv run pytest`, `uv run ruff check .`, ingest, eval — all pass/succeed

**Dependencies:** All previous tasks

**Files likely touched:**
- `README.md`

**Estimated scope:** Small

---

## Checkpoint: Complete
- [ ] All SPEC.md success criteria verified
- [ ] `uv run pytest` and `uv run ruff check .` pass clean
- [ ] Ready for human review / portfolio use
