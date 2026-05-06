import logging
import time
from functools import lru_cache

from fastapi import APIRouter, Request

from phonetics_engine.auth import require_internal_token
from phonetics_engine.config import Settings
from phonetics_engine.decision import classify
from phonetics_engine.enums import Decision, EntityType, MatchField
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.loader import CompanyRecord, EmployeeRecord, fetch_companies, fetch_employees
from phonetics_engine.matcher import (
    TenantIndex,
    build_company_index,
    build_employee_index,
)
from phonetics_engine.models import (
    CompanyCandidate,
    EmployeeCandidate,
    MatchRequest,
    MatchResponse,
    Thresholds,
)

logger = logging.getLogger(__name__)

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


def _search_candidates_override(req: MatchRequest, s: Settings):
    fields = _resolve_fields(req)
    if req.entity_type == EntityType.COMPANY:
        records = [
            CompanyRecord(
                id=c.id,
                display_name=c.display_name,
                canonical_name=c.canonical_name,
                aliases=c.aliases,
            )
            for c in req.candidates or []
            if isinstance(c, CompanyCandidate)
        ]
        index = build_company_index(records, match_fields=fields)
    else:
        records = [
            EmployeeRecord(
                id=c.id,
                first_name=c.first_name,
                infix=c.infix,
                last_name=c.last_name,
                full_name=c.full_name,
                company_ids=[],
            )
            for c in req.candidates or []
            if isinstance(c, EmployeeCandidate)
        ]
        index = build_employee_index(records, match_fields=fields)
    return index.search(req.query, top_k=req.top_k)


@router.post("/v1/match", response_model=MatchResponse, dependencies=[require_internal_token()])
async def match(req: MatchRequest, request: Request) -> MatchResponse:
    s = _settings()
    cache: IndexCache = request.app.state.index_cache
    thresholds = _resolve_thresholds(req, s)
    started = time.perf_counter()
    cache_status = "miss"

    try:
        if req.candidates is not None:
            scored = _search_candidates_override(req, s)
            cache_status = "n/a"
        else:
            existing = cache._values.get(_cache_key(req))  # noqa: SLF001
            if existing is not None and existing[0] > time.monotonic():
                cache_status = "hit"

            async def builder() -> TenantIndex:
                if req.entity_type == EntityType.COMPANY:
                    return await _build_company_tenant_index(req, s)
                return await _build_employee_tenant_index(req, s)

            index = await cache.get_or_build(_cache_key(req), builder)
            scored = index.search(req.query, top_k=req.top_k)

        decision, matches = classify(req.query, scored, thresholds, req.top_k)
    except Exception:
        logger.exception(
            "match_service_error customer_id=%s entity_type=%s",
            req.customer_id, req.entity_type.value,
        )
        return MatchResponse(
            entity_type=req.entity_type,
            decision=Decision.SERVICE_ERROR,
            applied_thresholds=thresholds,
            matches=[],
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    top_score = matches[0].score if matches else 0.0
    logger.info(
        "match customer_id=%s entity_type=%s decision=%s top_score=%.3f cache=%s latency_ms=%d",
        req.customer_id, req.entity_type.value, decision.value, top_score, cache_status, elapsed_ms,
    )

    return MatchResponse(
        entity_type=req.entity_type,
        decision=decision,
        applied_thresholds=thresholds,
        matches=matches,
    )
