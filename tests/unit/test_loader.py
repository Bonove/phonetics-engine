import httpx
import pytest
import respx

from phonetics_engine.config import Settings
from phonetics_engine.loader import (
    CompanyRecord,
    EmployeeRecord,
    fetch_companies,
    fetch_employees,
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "tok")
    return Settings()


@respx.mock
@pytest.mark.asyncio
async def test_fetch_companies_returns_records(settings):
    respx.get(
        "https://test.supabase.co/rest/v1/companies",
        params={"customer_id": "eq.1000435", "select": "id,display_name,canonical_name,aliases"},
    ).mock(return_value=httpx.Response(200, json=[
        {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        {
            "id": "c2",
            "display_name": "TaxiCentrale Maassluis",
            "canonical_name": "taxicentrale maassluis",
            "aliases": ["TCM"],
        },
    ]))

    out = await fetch_companies(settings, customer_id="1000435")
    assert len(out) == 2
    assert isinstance(out[0], CompanyRecord)
    assert out[0].canonical_name == "waysis"
    assert out[1].aliases == ["TCM"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_employees_filters_by_customer(settings):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "e1",
                "first_name": "Sanne",
                "infix": "de",
                "last_name": "Vries",
                "full_name": "Sanne de Vries",
                "company_id": "c1",
                "companies": {"customer_id": "1000435"},
            },
        ])
    )
    out = await fetch_employees(settings, customer_id="1000435")
    assert len(out) == 1
    assert isinstance(out[0], EmployeeRecord)
    assert out[0].full_name == "Sanne de Vries"
    assert out[0].company_id == "c1"

    request = respx.calls.last.request
    qs = request.url.params
    assert qs["companies.customer_id"] == "eq.1000435"
    assert "company_id" not in qs.multi_items() or all(
        k != "company_id" for k, _ in qs.multi_items()
    )
    assert "company_id" in qs["select"]
    assert "companies!inner(customer_id)" in qs["select"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_employees_filters_by_company_id(settings):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "A", "infix": None, "last_name": "X", "full_name": "A X",
             "company_id": "c1", "companies": {"customer_id": "1000435"}},
        ])
    )
    out = await fetch_employees(settings, customer_id="1000435", company_id="c1")
    assert {e.id for e in out} == {"e1"}

    request = respx.calls.last.request
    qs = request.url.params
    assert qs["companies.customer_id"] == "eq.1000435"
    assert qs["company_id"] == "eq.c1"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_companies_propagates_error(settings):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_companies(settings, customer_id="1000435")
