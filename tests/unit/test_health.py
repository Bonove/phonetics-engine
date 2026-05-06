from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


def test_health_returns_ok(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "t")
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
