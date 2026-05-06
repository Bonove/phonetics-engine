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
def test_customer_id_filter_is_passed_to_supabase(client):
    route_a = respx.get(
        "https://test.supabase.co/rest/v1/companies",
        params={"customer_id": "eq.tenant-a", "select": "id,display_name,canonical_name,aliases"},
    ).mock(return_value=httpx.Response(200, json=[
        {"id": "ca", "display_name": "Alpha", "canonical_name": "alpha", "aliases": []},
    ]))

    route_b = respx.get(
        "https://test.supabase.co/rest/v1/companies",
        params={"customer_id": "eq.tenant-b", "select": "id,display_name,canonical_name,aliases"},
    ).mock(return_value=httpx.Response(200, json=[
        {"id": "cb", "display_name": "Beta", "canonical_name": "beta", "aliases": []},
    ]))

    r_a = client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                     json={"query": "alpha", "entity_type": "company", "customer_id": "tenant-a"})
    r_b = client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                     json={"query": "beta", "entity_type": "company", "customer_id": "tenant-b"})

    assert route_a.called and route_b.called
    assert r_a.json()["matches"][0]["id"] == "ca"
    assert r_b.json()["matches"][0]["id"] == "cb"
