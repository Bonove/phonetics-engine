from phonetics_engine.enums import Decision, EntityType, MatchField


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
