from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from phonetics_engine.auth import require_internal_token
from phonetics_engine.enums import EntityType
from phonetics_engine.index_cache import IndexCache

router = APIRouter()


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str
    entity_type: EntityType | None = None


class ReloadResponse(BaseModel):
    flushed: bool
    customer_id: str
    entity_type: EntityType | None


@router.post("/v1/reload", response_model=ReloadResponse, dependencies=[require_internal_token()])
async def reload(req: ReloadRequest, request: Request) -> ReloadResponse:
    cache: IndexCache = request.app.state.index_cache
    if req.entity_type is None:
        cache.invalidate_prefix((req.customer_id,))
    else:
        cache.invalidate_prefix((req.customer_id, req.entity_type.value))
    return ReloadResponse(flushed=True, customer_id=req.customer_id, entity_type=req.entity_type)
