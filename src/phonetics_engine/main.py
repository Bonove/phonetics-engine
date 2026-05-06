from fastapi import FastAPI

from phonetics_engine.routes import health


def create_app() -> FastAPI:
    app = FastAPI(title="phonetics-engine", version="0.1.0")
    app.include_router(health.router)
    return app


app = create_app()
