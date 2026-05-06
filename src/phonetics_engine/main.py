import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.logging_setup import configure_logging
from phonetics_engine.prewarm import prewarm_all
from phonetics_engine.routes import health, legacy, match
from phonetics_engine.routes import metrics as metrics_route
from phonetics_engine.routes import reload as reload_route


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    cache: IndexCache = app.state.index_cache
    if settings.phx_prewarm_enabled:
        app.state.prewarm_task = asyncio.create_task(prewarm_all(settings, cache))
    yield
    task = getattr(app.state, "prewarm_task", None)
    if task and not task.done():
        task.cancel()


def create_app() -> FastAPI:
    configure_logging()
    settings = Settings()
    app = FastAPI(title="phonetics-engine", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.index_cache = IndexCache(ttl_seconds=float(settings.phx_cache_ttl_seconds))
    app.include_router(health.router)
    app.include_router(match.router)
    app.include_router(reload_route.router)
    app.include_router(metrics_route.router)
    app.include_router(legacy.router)
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
