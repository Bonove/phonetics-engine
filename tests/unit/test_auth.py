import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from phonetics_engine.auth import require_internal_token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret-token")

    app = FastAPI()

    @app.get("/protected", dependencies=[require_internal_token()])
    def protected():
        return {"ok": True}

    return TestClient(app)


def test_missing_header_returns_401(client):
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_token_returns_401(client):
    r = client.get("/protected", headers={"X-Internal-Token": "nope"})
    assert r.status_code == 401


def test_correct_token_passes(client):
    r = client.get("/protected", headers={"X-Internal-Token": "secret-token"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
