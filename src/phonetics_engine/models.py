from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from phonetics_engine.enums import Decision, EntityType, MatchField


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_match: Annotated[float, Field(ge=0.0, le=1.0)]
    high_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    ambiguity_margin: Annotated[float, Field(ge=0.0, le=1.0)]


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_id: str


class CompanyCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)


class EmployeeCandidate(BaseModel):
    # extra="forbid" rejects `phone` (and any other unexpected field) -> 422
    model_config = ConfigDict(extra="forbid")

    id: str
    first_name: str
    infix: str | None = None
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        if self.infix and self.infix.strip():
            return f"{self.first_name} {self.infix.strip()} {self.last_name}"
        return f"{self.first_name} {self.last_name}"


Candidate = CompanyCandidate | EmployeeCandidate


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    entity_type: EntityType
    customer_id: str
    scope: Scope | None = None
    match_fields: list[MatchField] | None = None
    thresholds: Thresholds | None = None
    candidates: list[Candidate] | None = None
    top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must be non-empty")
        return v.strip()


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    display_name: str
    canonical_name: str
    score: float
    margin_to_next: float
    matched_field: MatchField | None = None
    matched_value: str | None = None


class MatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    decision: Decision
    applied_thresholds: Thresholds
    matches: list[Match]
