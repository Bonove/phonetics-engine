import httpx
import pytest
import respx

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.prewarm import prewarm_all

pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "t")
    return Settings()


@respx.mock
@pytest.mark.asyncio
async def test_prewarm_lists_tenants_and_builds_indexes(settings):
    respx.get("https://test.supabase.co/rest/v1/customers").mock(
        return_value=httpx.Response(200, json=[{"id": "tenant-a"}, {"id": "tenant-b"}])
    )
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "X", "canonical_name": "x", "aliases": []},
        ])
    )
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "A", "infix": None, "last_name": "Z",
             "full_name": "A Z", "employee_company_roles": []},
        ])
    )

    cache = IndexCache(ttl_seconds=60.0)
    await prewarm_all(settings, cache)

    # Both tenants warmed for both entity types -> 4 cache entries
    assert len(cache._values) == 4  # noqa: SLF001
