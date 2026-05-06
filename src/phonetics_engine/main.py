from fastapi import FastAPI

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.routes import health, match


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="phonetics-engine", version="0.1.0")
    app.state.settings = settings
    app.state.index_cache = IndexCache(ttl_seconds=float(settings.phx_cache_ttl_seconds))
    app.include_router(health.router)
    app.include_router(match.router)
    return app


def _make_app() -> FastAPI:
    try:
        return create_app()
    except Exception:
        # During test collection Settings() may fail if env vars are not set.
        # The real app is created per-test via create_app() after monkeypatching.
        _stub = FastAPI(title="phonetics-engine", version="0.1.0")
        return _stub


app = _make_app()
