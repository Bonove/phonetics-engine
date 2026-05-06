import httpx
import pytest
import respx
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


@respx.mock
def test_supabase_5xx_becomes_200_service_error(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(500, text="boom")
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "company", "customer_id": "any"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "service_error"
    assert body["matches"] == []


@respx.mock
def test_empty_tenant_returns_200_no_match(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "company", "customer_id": "empty-tenant"},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "no_match"


def test_invalid_entity_type_returns_422(client):
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "vehicle", "customer_id": "x"},
    )
    assert r.status_code == 422


def test_missing_token_returns_401(client):
    r = client.post(
        "/v1/match",
        json={"query": "x", "entity_type": "company", "customer_id": "x"},
    )
    assert r.status_code == 401
