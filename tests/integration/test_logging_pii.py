import logging

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
    monkeypatch.setenv("PHX_LOG_PAYLOAD", "0")
    yield TestClient(create_app())
    auth._settings.cache_clear()
    match_route._settings.cache_clear()


@respx.mock
def test_info_log_contains_no_pii(client, caplog):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])
    )
    with caplog.at_level(logging.INFO, logger="phonetics_engine"):
        client.post(
            "/v1/match",
            headers={"X-Internal-Token": "secret"},
            json={"query": "Wasteless Inc", "entity_type": "company", "customer_id": "tenant-x"},
        )

    text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Wasteless Inc" not in text  # query: PII
    assert "Waysis" not in text  # display_name: PII
    assert "secret" not in text  # never log the token
    assert "tenant-x" in text  # customer_id is fine
    assert "single_high_confidence" in text or "no_match" in text or "exact" in text
