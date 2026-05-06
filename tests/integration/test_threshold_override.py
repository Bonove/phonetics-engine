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
def test_request_thresholds_override_env_defaults(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "wasteless",
            "entity_type": "company",
            "customer_id": "x",
            "thresholds": {"min_match": 0.10, "high_confidence": 0.20, "ambiguity_margin": 0.01},
        },
    )
    body = r.json()
    assert body["applied_thresholds"]["min_match"] == 0.10
    assert body["applied_thresholds"]["high_confidence"] == 0.20
    assert body["applied_thresholds"]["ambiguity_margin"] == 0.01


@respx.mock
def test_default_thresholds_use_env_for_employee(client):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": None, "last_name": "Vries",
             "full_name": "Sanne Vries", "employee_company_roles": []},
        ])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "Vries", "entity_type": "employee", "customer_id": "x"},
    )
    body = r.json()
    assert body["applied_thresholds"]["min_match"] == 0.55
    assert body["applied_thresholds"]["high_confidence"] == 0.86  # employee default
    assert body["applied_thresholds"]["ambiguity_margin"] == 0.12
