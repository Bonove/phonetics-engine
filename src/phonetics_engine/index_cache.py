import asyncio
import time
from collections.abc import Awaitable, Callable, Hashable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class IndexCache(Generic[T]):  # noqa: UP046
    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._values: dict[Hashable, tuple[float, T]] = {}
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _is_fresh(self, expires_at: float) -> bool:
        return time.monotonic() < expires_at

    async def _key_lock(self, key: Hashable) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def get_or_build(self, key: Hashable, builder: Callable[[], Awaitable[T]]) -> T:
        entry = self._values.get(key)
        if entry is not None and self._is_fresh(entry[0]):
            return entry[1]

        lock = await self._key_lock(key)
        async with lock:
            entry = self._values.get(key)
            if entry is not None and self._is_fresh(entry[0]):
                return entry[1]
            built = await builder()
            self._values[key] = (time.monotonic() + self._ttl, built)
            return built

    def invalidate(self, key: Hashable) -> None:
        self._values.pop(key, None)

    def invalidate_prefix(self, prefix: tuple[Any, ...]) -> None:
        plen = len(prefix)
        to_drop = [
            k for k in list(self._values.keys())
            if isinstance(k, tuple) and len(k) >= plen and k[:plen] == prefix
        ]
        for k in to_drop:
            self._values.pop(k, None)
