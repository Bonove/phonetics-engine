import asyncio
import time

import httpx
import pytest
import respx

from phonetics_engine.main import create_app

pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    yield create_app()
    import phonetics_engine.auth as _auth
    import phonetics_engine.routes.match as _match
    _auth._settings.cache_clear()
    _match._settings.cache_clear()


@respx.mock
@pytest.mark.asyncio
async def test_parallel_misses_call_supabase_once(app):
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])

    respx.get("https://test.supabase.co/rest/v1/companies").mock(side_effect=handler)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        results = await asyncio.gather(*(
            client.post(
                "/v1/match",
                headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "tenant-x"},
            )
            for _ in range(5)
        ))
    assert all(r.status_code == 200 for r in results)
    assert call_count == 1
