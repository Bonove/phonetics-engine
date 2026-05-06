import pytest
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    yield TestClient(create_app())
    import phonetics_engine.auth as _auth
    import phonetics_engine.routes.match as _match
    _auth._settings.cache_clear()
    _match._settings.cache_clear()


def test_employee_candidate_with_phone_returns_422(client):
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Vries",
            "entity_type": "employee",
            "customer_id": "x",
            "candidates": [
                {"id": "e1", "first_name": "Sanne", "last_name": "Vries", "phone": "31600000000"},
            ],
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert "phone" in str(body).lower()
