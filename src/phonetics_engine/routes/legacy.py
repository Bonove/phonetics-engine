from functools import lru_cache

import httpx
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from phonetics_engine.config import Settings
from phonetics_engine.loader import _headers
from phonetics_engine.phonetics import PhoneticIndex, phonemize_name

router = APIRouter()


class LegacyRequest(BaseModel):
    name: str
    top_k: int = 3
    min_score: float = 0.3


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


async def _fetch_legacy_rows() -> list[dict]:
    s = _settings()
    url = f"{s.supabase_url}/rest/v1/medewerkers_bellijst"
    params = {"select": "voornaam,company_name,telefoonnummer,id"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers=_headers(s))
        r.raise_for_status()
        return r.json()


@router.post("/search")
async def legacy_search(
    req: LegacyRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    rows = await _fetch_legacy_rows()
    names = [row["voornaam"] for row in rows]
    metadata = [{"company": row["company_name"], "phone": row["telefoonnummer"]} for row in rows]

    index = PhoneticIndex(names)
    raw = index.search(req.name, top_k=req.top_k)
    out = []
    for r in raw:
        if r["score"] < req.min_score:
            continue
        meta = metadata[names.index(r["name"])]
        out.append({
            "name": r["name"],
            "score": r["score"],
            "phonemes": r["phonemes"],
            "company": meta["company"],
            "phone": meta["phone"],
        })
    return {
        "matches": out,
        "query_phonemes": phonemize_name(req.name),
        "source": "supabase",
    }
