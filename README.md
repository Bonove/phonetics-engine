# phonetics-engine

Multi-tenant phonetic matching service. See `GOAL.md` for the full spec and `docs/superpowers/plans/` for the implementation plan.

## Local dev

    uv sync --extra dev
    cp .env.example .env  # fill in real values
    uv run uvicorn phonetics_engine.main:app --reload

## Tests

    uv run pytest -v
