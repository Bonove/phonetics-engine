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


def _stub_supabase():
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": "de", "last_name": "Vries",
             "full_name": "Sanne de Vries", "company_id": "c1"},
        ])
    )


@respx.mock
def test_query_last_name_only(client):
    _stub_supabase()
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "Vries", "entity_type": "employee", "customer_id": "t"},
    )
    body = r.json()
    assert body["matches"][0]["id"] == "e1"
    assert body["matches"][0]["matched_field"] == "last_name"
    assert body["matches"][0]["matched_value"] == "Vries"


@respx.mock
def test_query_with_infix(client):
    _stub_supabase()
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "de Vries", "entity_type": "employee", "customer_id": "t"},
    )
    body = r.json()
    assert body["matches"][0]["id"] == "e1"
    assert body["matches"][0]["matched_field"] == "last_name_with_infix"
    assert body["matches"][0]["matched_value"] == "de Vries"
