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
    from phonetics_engine.routes import legacy as legacy_route
    from phonetics_engine.routes import match as match_route

    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    yield TestClient(create_app())
    auth._settings.cache_clear()
    match_route._settings.cache_clear()
    legacy_route._settings.cache_clear()


@respx.mock
def test_legacy_search_returns_bonove_shape(client):
    respx.get("https://test.supabase.co/rest/v1/medewerkers_bellijst").mock(
        return_value=httpx.Response(200, json=[
            {"id": "1", "voornaam": "Max", "telefoonnummer": "31621449795",
             "company_name": "waysis"},
            {"id": "2", "voornaam": "Steven", "telefoonnummer": "31621200435",
             "company_name": "tmc"},
        ])
    )

    r = client.post(
        "/search",
        headers={"Authorization": "Bearer legacy-token"},
        json={"name": "Max", "top_k": 3, "min_score": 0.3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "matches" in body
    assert "query_phonemes" in body
    assert body["source"] == "supabase"
    assert body["matches"][0]["name"] == "Max"
    assert body["matches"][0]["company"] == "waysis"
    assert body["matches"][0]["phone"] == "31621449795"
