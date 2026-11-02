# syntax=docker/dockerfile:1.7
# Multi-stage build: uv installs locked deps (CPU-only torch) in a builder; runtime is slim + non-root.
# The embedding model is NOT baked in: mount the Hugging Face cache as a volume (see README).

# ---------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached layer), from the lock file only: --frozen fails if the lock is stale.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app data/records ./data/records

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=uat \
    LOG_LEVEL=INFO \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False \
    ANSWER_PROVIDER=none \
    CHROMA_PATH=/app/data/chroma_uat \
    RECORDS_DIR=/app/data/records

RUN mkdir -p /app/data/chroma_uat /home/app/.cache/huggingface \
 && chown -R app:app /app/data /home/app/.cache

USER app

EXPOSE 8000

# Model load takes a few seconds on start; health returns 503 until it is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
