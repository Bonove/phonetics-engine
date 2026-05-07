from enum import StrEnum


class EntityType(StrEnum):
    COMPANY = "company"
    EMPLOYEE = "employee"


class Decision(StrEnum):
    EXACT = "exact"
    SINGLE_HIGH_CONFIDENCE = "single_high_confidence"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no_match"
    SERVICE_ERROR = "service_error"


class MatchField(StrEnum):
    DISPLAY_NAME = "display_name"
    CANONICAL_NAME = "canonical_name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    LAST_NAME_WITH_INFIX = "last_name_with_infix"
    FULL_NAME = "full_name"
    ALIAS = "alias"
