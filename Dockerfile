FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Deps installed from the lockfile before copying source so this layer is
# cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uvicorn", "rag_project.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
