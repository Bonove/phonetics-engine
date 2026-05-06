FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libespeak-ng1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "phonetics_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
