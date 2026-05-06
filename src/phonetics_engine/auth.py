from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from phonetics_engine.config import Settings


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


def _verify(x_internal_token: str | None = Header(default=None)) -> None:
    expected = _settings().phx_internal_token
    if not x_internal_token or x_internal_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")


def require_internal_token() -> Depends:
    return Depends(_verify)
