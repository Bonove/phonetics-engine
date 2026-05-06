import pytest
from pydantic import ValidationError

from phonetics_engine.enums import Decision, EntityType, MatchField
from phonetics_engine.models import (
    CompanyCandidate,
    EmployeeCandidate,
    Match,
    MatchRequest,
    MatchResponse,
    Scope,
    Thresholds,
)


def test_entity_type_values():
    assert EntityType.COMPANY.value == "company"
    assert EntityType.EMPLOYEE.value == "employee"


def test_decision_values():
    assert Decision.EXACT.value == "exact"
    assert Decision.SINGLE_HIGH_CONFIDENCE.value == "single_high_confidence"
    assert Decision.AMBIGUOUS.value == "ambiguous"
    assert Decision.NO_MATCH.value == "no_match"
    assert Decision.SERVICE_ERROR.value == "service_error"


def test_match_field_values():
    assert MatchField.DISPLAY_NAME.value == "display_name"
    assert MatchField.CANONICAL_NAME.value == "canonical_name"
    assert MatchField.LAST_NAME.value == "last_name"
    assert MatchField.LAST_NAME_WITH_INFIX.value == "last_name_with_infix"
    assert MatchField.FULL_NAME.value == "full_name"
    assert MatchField.ALIAS.value == "alias"


def test_thresholds_validation():
    t = Thresholds(min_match=0.5, high_confidence=0.8, ambiguity_margin=0.1)
    assert t.min_match == 0.5

    with pytest.raises(ValidationError):
        Thresholds(min_match=1.5, high_confidence=0.8, ambiguity_margin=0.1)


def test_company_candidate_basic():
    c = CompanyCandidate(id="c1", display_name="Waysis", canonical_name="waysis")
    assert c.id == "c1"
    assert c.aliases == []


def test_employee_candidate_forbids_phone():
    with pytest.raises(ValidationError) as exc:
        EmployeeCandidate(
            id="e1",
            first_name="Sanne",
            last_name="de Vries",
            phone="31600000000",
        )
    assert "phone" in str(exc.value).lower()


def test_employee_candidate_minimal():
    c = EmployeeCandidate(id="e1", first_name="Sanne", last_name="Vries")
    assert c.full_name == "Sanne Vries"
    assert c.infix is None


def test_employee_candidate_with_infix():
    c = EmployeeCandidate(id="e1", first_name="Sanne", infix="de", last_name="Vries")
    assert c.full_name == "Sanne de Vries"


def test_match_request_company_minimal():
    r = MatchRequest(
        query="wasteless",
        entity_type=EntityType.COMPANY,
        customer_id="1000435",
    )
    assert r.scope is None
    assert r.candidates is None
    assert r.thresholds is None
    assert r.match_fields is None
    assert r.top_k == 5


def test_match_request_employee_with_scope():
    r = MatchRequest(
        query="vries",
        entity_type=EntityType.EMPLOYEE,
        customer_id="1000435",
        scope=Scope(company_id="11111111-1111-1111-1111-111111111111"),
        match_fields=[MatchField.LAST_NAME, MatchField.FULL_NAME],
    )
    assert r.scope.company_id == "11111111-1111-1111-1111-111111111111"


def test_match_request_rejects_unknown_entity_type():
    with pytest.raises(ValidationError):
        MatchRequest(query="x", entity_type="vehicle", customer_id="x")


def test_match_response_no_phone_in_match():
    m = Match(
        id="e1",
        display_name="Sanne de Vries",
        canonical_name="sanne de vries",
        score=0.78,
        margin_to_next=0.04,
        matched_field=MatchField.LAST_NAME,
        matched_value="de Vries",
    )
    dumped = m.model_dump()
    assert "phone" not in dumped


def test_match_response_serializes():
    resp = MatchResponse(
        entity_type=EntityType.COMPANY,
        decision=Decision.SINGLE_HIGH_CONFIDENCE,
        applied_thresholds=Thresholds(min_match=0.55, high_confidence=0.82, ambiguity_margin=0.10),
        matches=[
            Match(
                id="c1",
                display_name="Waysis",
                canonical_name="waysis",
                score=0.91,
                margin_to_next=0.40,
            )
        ],
    )
    j = resp.model_dump_json()
    assert '"entity_type":"company"' in j
    assert '"decision":"single_high_confidence"' in j
