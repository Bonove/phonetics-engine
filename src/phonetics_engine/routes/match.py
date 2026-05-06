from functools import lru_cache

from fastapi import APIRouter, Request

from phonetics_engine.auth import require_internal_token
from phonetics_engine.config import Settings
from phonetics_engine.decision import classify
from phonetics_engine.enums import EntityType, MatchField
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.loader import fetch_companies, fetch_employees
from phonetics_engine.matcher import (
    TenantIndex,
    build_company_index,
    build_employee_index,
)
from phonetics_engine.models import MatchRequest, MatchResponse, Thresholds

router = APIRouter()


_DEFAULT_FIELDS_COMPANY = [MatchField.DISPLAY_NAME, MatchField.CANONICAL_NAME, MatchField.ALIAS]
_DEFAULT_FIELDS_EMPLOYEE = [MatchField.LAST_NAME, MatchField.FULL_NAME]


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


def _company_thresholds(s: Settings) -> Thresholds:
    return Thresholds(
        min_match=s.phx_company_min_match,
        high_confidence=s.phx_company_high_confidence,
        ambiguity_margin=s.phx_company_ambiguity_margin,
    )


def _employee_thresholds(s: Settings) -> Thresholds:
    return Thresholds(
        min_match=s.phx_employee_min_match,
        high_confidence=s.phx_employee_high_confidence,
        ambiguity_margin=s.phx_employee_ambiguity_margin,
    )


def _resolve_thresholds(req: MatchRequest, s: Settings) -> Thresholds:
    if req.thresholds is not None:
        return req.thresholds
    if req.entity_type == EntityType.COMPANY:
        return _company_thresholds(s)
    return _employee_thresholds(s)


def _resolve_fields(req: MatchRequest) -> list[MatchField]:
    if req.match_fields:
        return req.match_fields
    if req.entity_type == EntityType.COMPANY:
        return _DEFAULT_FIELDS_COMPANY
    return _DEFAULT_FIELDS_EMPLOYEE


async def _build_company_tenant_index(req: MatchRequest, s: Settings) -> TenantIndex:
    records = await fetch_companies(s, customer_id=req.customer_id)
    return build_company_index(records, match_fields=_resolve_fields(req))


async def _build_employee_tenant_index(req: MatchRequest, s: Settings) -> TenantIndex:
    company_id = req.scope.company_id if req.scope else None
    records = await fetch_employees(s, customer_id=req.customer_id, company_id=company_id)
    return build_employee_index(records, match_fields=_resolve_fields(req))


def _cache_key(req: MatchRequest) -> tuple:
    if req.entity_type == EntityType.COMPANY:
        return (req.customer_id, "company", tuple(_resolve_fields(req)))
    cid = req.scope.company_id if req.scope else None
    return (req.customer_id, "employee", cid, tuple(_resolve_fields(req)))


@router.post("/v1/match", response_model=MatchResponse, dependencies=[require_internal_token()])
async def match(req: MatchRequest, request: Request) -> MatchResponse:
    s = _settings()
    cache: IndexCache = request.app.state.index_cache
    thresholds = _resolve_thresholds(req, s)

    async def builder() -> TenantIndex:
        if req.entity_type == EntityType.COMPANY:
            return await _build_company_tenant_index(req, s)
        return await _build_employee_tenant_index(req, s)

    index = await cache.get_or_build(_cache_key(req), builder)
    scored = index.search(req.query, top_k=req.top_k)
    decision, matches = classify(req.query, scored, thresholds, req.top_k)

    return MatchResponse(
        entity_type=req.entity_type,
        decision=decision,
        applied_thresholds=thresholds,
        matches=matches,
    )
