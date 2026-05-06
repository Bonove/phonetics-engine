import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app

pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    from phonetics_engine import auth
    from phonetics_engine.routes import match as match_route

    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    yield TestClient(create_app())
    auth._settings.cache_clear()
    match_route._settings.cache_clear()


@respx.mock
def test_reload_invalidates_specific_entity(client):
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])

    respx.get("https://test.supabase.co/rest/v1/companies").mock(side_effect=handler)

    # First call -> miss -> Supabase hit (call 1)
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    # Second call -> hit -> no Supabase
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    assert call_count == 1

    # Reload company-only
    r = client.post("/v1/reload",
                    headers={"X-Internal-Token": "secret"},
                    json={"customer_id": "t", "entity_type": "company"})
    assert r.status_code == 200
    assert r.json() == {"flushed": True, "customer_id": "t", "entity_type": "company"}

    # Third call -> miss again -> Supabase hit (call 2)
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    assert call_count == 2


def test_reload_requires_token(client):
    r = client.post("/v1/reload", json={"customer_id": "x"})
    assert r.status_code == 401


def test_reload_customer_wide_when_entity_type_null(client):
    r = client.post("/v1/reload", headers={"X-Internal-Token": "secret"},
                    json={"customer_id": "t"})
    assert r.status_code == 200
    assert r.json() == {"flushed": True, "customer_id": "t", "entity_type": None}
