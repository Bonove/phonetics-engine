from dataclasses import dataclass

import httpx

from phonetics_engine.config import Settings


@dataclass(frozen=True)
class CompanyRecord:
    id: str
    display_name: str
    canonical_name: str
    aliases: list[str]


@dataclass(frozen=True)
class EmployeeRecord:
    id: str
    first_name: str
    infix: str | None
    last_name: str
    full_name: str
    company_ids: list[str]


def _headers(settings: Settings) -> dict[str, str]:
    return {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
        "Accept": "application/json",
    }


async def fetch_companies(settings: Settings, *, customer_id: str) -> list[CompanyRecord]:
    url = f"{settings.supabase_url}/rest/v1/companies"
    params = {
        "customer_id": f"eq.{customer_id}",
        "select": "id,display_name,canonical_name,aliases",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers=_headers(settings))
        r.raise_for_status()
        return [
            CompanyRecord(
                id=row["id"],
                display_name=row["display_name"],
                canonical_name=row["canonical_name"],
                aliases=list(row.get("aliases") or []),
            )
            for row in r.json()
        ]


async def fetch_employees(
    settings: Settings,
    *,
    customer_id: str,
    company_id: str | None = None,
) -> list[EmployeeRecord]:
    url = f"{settings.supabase_url}/rest/v1/employees"
    params = {
        "customer_id": f"eq.{customer_id}",
        "select": "id,first_name,infix,last_name,full_name,employee_company_roles(company_id)",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers=_headers(settings))
        r.raise_for_status()
        rows = r.json()

    out: list[EmployeeRecord] = []
    for row in rows:
        company_ids = [r["company_id"] for r in row.get("employee_company_roles") or []]
        if company_id is not None and company_id not in company_ids:
            continue
        out.append(
            EmployeeRecord(
                id=row["id"],
                first_name=row["first_name"],
                infix=row.get("infix"),
                last_name=row["last_name"],
                full_name=row["full_name"],
                company_ids=company_ids,
            )
        )
    return out
