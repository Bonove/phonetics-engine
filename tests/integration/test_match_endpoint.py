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
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    app = create_app()
    yield TestClient(app)
    # Clear lru_caches so later tests don't see stale Settings from our monkeypatched env.
    import phonetics_engine.auth as _auth
    import phonetics_engine.routes.match as _match
    _auth._settings.cache_clear()
    _match._settings.cache_clear()


@respx.mock
def test_match_company_single_high_confidence(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
            {"id": "c2", "display_name": "Wasteless", "canonical_name": "wasteless", "aliases": []},
        ])
    )

    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "waysis", "entity_type": "company", "customer_id": "xpots-dev"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["entity_type"] == "company"
    assert body["decision"] in ("exact", "single_high_confidence")
    assert body["matches"][0]["id"] == "c1"
    assert "phone" not in body["matches"][0]


@respx.mock
def test_match_employee_with_scope_filters(client):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": "de", "last_name": "Vries",
             "full_name": "Sanne de Vries", "employee_company_roles": [{"company_id": "c1"}]},
            {"id": "e2", "first_name": "Bert", "infix": None, "last_name": "Jansen",
             "full_name": "Bert Jansen", "employee_company_roles": [{"company_id": "c2"}]},
        ])
    )

    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Vries",
            "entity_type": "employee",
            "customer_id": "xpots-dev",
            "scope": {"company_id": "c1"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["matches"][0]["id"] == "e1"


def test_match_requires_token(client):
    r = client.post("/v1/match", json={"query": "x", "entity_type": "company", "customer_id": "x"})
    assert r.status_code == 401


def test_candidates_override_skips_db(client):
    # No respx stubs registered — if the loader is called, this would fail.
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Waysis",
            "entity_type": "company",
            "customer_id": "any",
            "candidates": [
                {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis"},
                {"id": "c2", "display_name": "Wasteless", "canonical_name": "wasteless"},
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["matches"][0]["id"] == "c1"
