import pytest

from phonetics_engine.enums import MatchField
from phonetics_engine.loader import CompanyRecord, EmployeeRecord
from phonetics_engine.matcher import (
    build_company_index,
    build_employee_index,
)

pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


def _company(id_, display, canonical, aliases=None):
    return CompanyRecord(
        id=id_, display_name=display, canonical_name=canonical, aliases=aliases or []
    )


def _employee(id_, first, infix, last):
    full = f"{first} {infix + ' ' if infix else ''}{last}".strip()
    return EmployeeRecord(
        id=id_, first_name=first, infix=infix, last_name=last, full_name=full, company_ids=[]
    )


def test_company_index_matches_canonical_name():
    idx = build_company_index([
        _company("c1", "Waysis", "waysis"),
        _company("c2", "Wasteless", "wasteless"),
    ], match_fields=[MatchField.DISPLAY_NAME, MatchField.CANONICAL_NAME])

    out = idx.search("waysis", top_k=2)
    assert out[0].id == "c1"
    assert out[0].score > 0.9
    assert out[0].matched_field in (MatchField.DISPLAY_NAME, MatchField.CANONICAL_NAME)


def test_company_index_uses_aliases():
    idx = build_company_index([
        _company("c1", "TaxiCentrale Maassluis", "taxicentrale maassluis", aliases=["TCM"]),
    ], match_fields=[MatchField.DISPLAY_NAME, MatchField.CANONICAL_NAME, MatchField.ALIAS])
    out = idx.search("TCM", top_k=2)
    assert out[0].id == "c1"
    assert out[0].matched_field == MatchField.ALIAS
    assert out[0].matched_value == "TCM"


def test_employee_index_matches_last_name():
    idx = build_employee_index([
        _employee("e1", "Sanne", "de", "Vries"),
        _employee("e2", "Bert", None, "Jansen"),
    ], match_fields=[MatchField.LAST_NAME, MatchField.FULL_NAME])

    out = idx.search("Vries", top_k=5)
    assert out[0].id == "e1"
    assert out[0].matched_field in (MatchField.LAST_NAME, MatchField.LAST_NAME_WITH_INFIX)


def test_employee_index_matches_infix_plus_last_name():
    idx = build_employee_index([
        _employee("e1", "Sanne", "de", "Vries"),
    ], match_fields=[MatchField.LAST_NAME, MatchField.FULL_NAME])

    out_plain = idx.search("Vries", top_k=1)
    out_infix = idx.search("de Vries", top_k=1)
    assert out_plain[0].matched_value == "Vries"
    assert out_infix[0].matched_value == "de Vries"
    assert out_infix[0].matched_field == MatchField.LAST_NAME_WITH_INFIX


def test_employee_dedup_keeps_highest_field_score():
    """A record may match on multiple fields; only the best should be returned."""
    idx = build_employee_index([
        _employee("e1", "Sanne", None, "Vries"),
    ], match_fields=[MatchField.LAST_NAME, MatchField.FULL_NAME])
    out = idx.search("Sanne Vries", top_k=5)
    ids = [c.id for c in out]
    assert ids.count("e1") == 1


def test_empty_index_returns_empty():
    idx = build_company_index([], match_fields=[MatchField.CANONICAL_NAME])
    assert idx.search("anything", top_k=5) == []
