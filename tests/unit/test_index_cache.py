import asyncio

import pytest

from phonetics_engine.index_cache import IndexCache


@pytest.mark.asyncio
async def test_cache_miss_calls_loader_once():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return f"index-{calls}"

    cache = IndexCache(ttl_seconds=60.0)
    out = await cache.get_or_build("k1", loader)
    assert out == "index-1"
    out2 = await cache.get_or_build("k1", loader)
    assert out2 == "index-1"  # cached
    assert calls == 1


@pytest.mark.asyncio
async def test_parallel_misses_call_loader_once():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "the-index"

    cache = IndexCache(ttl_seconds=60.0)
    results = await asyncio.gather(*(cache.get_or_build("k1", loader) for _ in range(5)))
    assert all(r == "the-index" for r in results)
    assert calls == 1


@pytest.mark.asyncio
async def test_ttl_expiry_triggers_rebuild():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return f"v{calls}"

    cache = IndexCache(ttl_seconds=0.05)
    a = await cache.get_or_build("k", loader)
    assert a == "v1"
    await asyncio.sleep(0.07)
    b = await cache.get_or_build("k", loader)
    assert b == "v2"
    assert calls == 2


@pytest.mark.asyncio
async def test_invalidate_drops_entry():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return f"v{calls}"

    cache = IndexCache(ttl_seconds=60.0)
    await cache.get_or_build("k", loader)
    cache.invalidate("k")
    await cache.get_or_build("k", loader)
    assert calls == 2


@pytest.mark.asyncio
async def test_invalidate_prefix():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return calls

    cache = IndexCache(ttl_seconds=60.0)
    await cache.get_or_build(("cust", "company"), loader)
    await cache.get_or_build(("cust", "employee"), loader)
    cache.invalidate_prefix(("cust",))
    await cache.get_or_build(("cust", "company"), loader)
    assert calls == 3
