import asyncio
import logging

import httpx

from phonetics_engine.config import Settings
from phonetics_engine.enums import MatchField
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.loader import _headers, fetch_companies, fetch_employees
from phonetics_engine.matcher import build_company_index, build_employee_index

logger = logging.getLogger(__name__)


_DEFAULT_FIELDS_COMPANY = (MatchField.DISPLAY_NAME, MatchField.CANONICAL_NAME, MatchField.ALIAS)
_DEFAULT_FIELDS_EMPLOYEE = (MatchField.LAST_NAME, MatchField.FULL_NAME)


async def _list_customer_ids(settings: Settings) -> list[str]:
    url = f"{settings.supabase_url}/rest/v1/customers"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params={"select": "id"}, headers=_headers(settings))
        r.raise_for_status()
        return [row["id"] for row in r.json()]


async def _warm_company(settings: Settings, cache: IndexCache, customer_id: str) -> None:
    fields = list(_DEFAULT_FIELDS_COMPANY)
    key = (customer_id, "company", tuple(fields))

    async def builder():
        records = await fetch_companies(settings, customer_id=customer_id)
        return build_company_index(records, match_fields=fields)

    try:
        await cache.get_or_build(key, builder)
    except Exception:
        logger.exception("prewarm_company_failed customer_id=%s", customer_id)


async def _warm_employee(settings: Settings, cache: IndexCache, customer_id: str) -> None:
    fields = list(_DEFAULT_FIELDS_EMPLOYEE)
    key = (customer_id, "employee", None, tuple(fields))

    async def builder():
        records = await fetch_employees(settings, customer_id=customer_id)
        return build_employee_index(records, match_fields=fields)

    try:
        await cache.get_or_build(key, builder)
    except Exception:
        logger.exception("prewarm_employee_failed customer_id=%s", customer_id)


async def prewarm_all(settings: Settings, cache: IndexCache) -> None:
    try:
        customer_ids = await _list_customer_ids(settings)
    except Exception:
        logger.exception("prewarm_list_customers_failed")
        return
    tasks = []
    for cid in customer_ids:
        tasks.append(_warm_company(settings, cache, cid))
        tasks.append(_warm_employee(settings, cache, cid))
    await asyncio.gather(*tasks)
    logger.info("prewarm_done tenants=%d", len(customer_ids))
