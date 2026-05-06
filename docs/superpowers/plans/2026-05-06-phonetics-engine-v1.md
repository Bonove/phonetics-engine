# phonetics-engine v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-tenant FastAPI service that matches spoken company- and employee-names against a tenant-scoped Supabase DB, returning a server-classified decision (`exact` / `single_high_confidence` / `ambiguous` / `no_match` / `service_error`) so Carla (Pipecat intercom bot) can route deterministically without client-side threshold logic.

**Architecture:** FastAPI + Pydantic v2 + phonemizer (espeak-ng) + FAISS (cosine via inner-product on unit-vectors). Supabase Postgres is the single source of truth (4-table schema: `customers`, `companies`, `employees`, `employee_company_roles`). Per-tenant FAISS indexes are cached in-process with a 60s TTL and an `asyncio.Lock` per cache-key against thundering-herd; a startup background task pre-warms known tenants. The legacy `/search` endpoint of Bonove's service stays untouched as a backwards-compatibility shim.

**Tech Stack:** Python 3.12+, uv, FastAPI, uvicorn[standard], Pydantic v2 + pydantic-settings, httpx (Supabase REST), numpy, faiss-cpu, phonemizer (espeak-ng system pkg), prometheus-client, pytest + pytest-asyncio + respx, ruff.

---

## Reference: GOAL.md and migrations/001_initial_schema.sql

Read these two files before starting any phase:

- `GOAL.md` — full functional spec (decisions, thresholds, status-code policy, logging discipline, SLOs).
- `migrations/001_initial_schema.sql` — DB schema this service queries.

The reused phonetic core (`phonemize_name`, `phonemize_batch`, `_phonemes_to_vector`, `PhoneticIndex.search`) is taken verbatim from Bonove's `phonetics-service/phonetics.py` and reproduced in Phase 4. No external dependency on the Bonove repo at runtime.

---

## File Structure

```
phonetics-engine/
├── pyproject.toml
├── Dockerfile
├── render.yaml
├── .env.example
├── .gitignore
├── README.md
├── migrations/
│   └── 001_initial_schema.sql        (already exists)
├── docs/superpowers/plans/
│   └── 2026-05-06-phonetics-engine-v1.md   (this file)
├── src/phonetics_engine/
│   ├── __init__.py
│   ├── main.py                       # uvicorn entrypoint + app factory
│   ├── config.py                     # Settings (pydantic-settings)
│   ├── enums.py                      # EntityType, Decision, MatchField
│   ├── models.py                     # MatchRequest, MatchResponse, Thresholds, Match, Scope, Candidate-shapes
│   ├── auth.py                       # X-Internal-Token FastAPI dependency
│   ├── normalize.py                  # canonicalize() — NFKD-lower-strip
│   ├── phonetics.py                  # phonemize_name/_batch + _phonemes_to_vector + PhoneticIndex
│   ├── decision.py                   # classify(query, matches, thresholds) -> (Decision, list[Match])
│   ├── loader.py                     # async Supabase REST queries
│   ├── matcher.py                    # build per-tenant index from records; multi-vector for employees
│   ├── index_cache.py                # async TTL cache with per-key Lock
│   ├── prewarm.py                    # startup background task
│   ├── logging_setup.py              # PII-aware stdlib logging config
│   ├── metrics.py                    # prometheus_client counters/histograms
│   └── routes/
│       ├── __init__.py
│       ├── health.py                 # GET /health (no PII)
│       ├── match.py                  # POST /v1/match
│       ├── reload.py                 # POST /v1/reload
│       ├── legacy.py                 # POST /search (Bonove shim)
│       └── metrics.py                # GET /metrics (Prometheus exposition)
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── unit/
    │   ├── test_normalize.py
    │   ├── test_decision.py
    │   ├── test_phonetics.py
    │   ├── test_matcher.py
    │   ├── test_index_cache.py
    │   ├── test_loader.py
    │   ├── test_models.py
    │   └── test_auth.py
    └── integration/
        ├── test_match_endpoint.py
        ├── test_reload_endpoint.py
        ├── test_legacy_search.py
        ├── test_tenant_isolation.py
        ├── test_concurrency_lock.py
        ├── test_phone_strip.py
        ├── test_threshold_override.py
        ├── test_status_policy.py
        ├── test_tussenvoegsel_matching.py
        ├── test_prewarm.py
        └── test_logging_pii.py
```

Each file has a single responsibility; no helper grab-bag files.

---

## Phase 0 — Project scaffold

**Goal:** Empty project becomes a runnable FastAPI app with `GET /health` returning `{"status": "ok"}`, a passing test, ruff clean, committed in git.

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `Dockerfile`, `render.yaml`
- Create: `src/phonetics_engine/__init__.py`, `src/phonetics_engine/main.py`, `src/phonetics_engine/config.py`, `src/phonetics_engine/routes/__init__.py`, `src/phonetics_engine/routes/health.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/unit/__init__.py`, `tests/unit/test_health.py`

### Task 0.1: Initialize git + uv project

- [ ] **Step 1: Init git**

```bash
cd /Users/tristanvandoorn@makerlab.nl/Documents/phonetics-engine
git init
git add GOAL.md migrations/ docs/
git commit -m "chore: import GOAL, migration, plan"
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.env
.env.local
dist/
*.egg-info/
.coverage
htmlcov/
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "phonetics-engine"
version = "0.1.0"
description = "Multi-tenant phonetic matching service for Carla."
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.7",
  "pydantic-settings>=2.4",
  "httpx>=0.27",
  "numpy>=1.26",
  "faiss-cpu>=1.8",
  "phonemizer>=3.3",
  "prometheus-client>=0.20",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-asyncio>=0.23",
  "respx>=0.21",
  "ruff>=0.6",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/phonetics_engine"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "ASYNC"]
```

- [ ] **Step 4: Run uv sync**

Run: `uv sync --extra dev`
Expected: virtual env at `.venv/`, all deps installed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: pyproject + uv lockfile"
```

(`uv.lock` will be created by `uv sync` and should also be staged.)

### Task 0.2: Settings module (pydantic-settings)

- [ ] **Step 1: Write failing test `tests/unit/test_config.py`**

```python
from phonetics_engine.config import Settings


def test_settings_defaults_load_from_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "test-token")
    s = Settings()
    assert s.supabase_url == "https://test.supabase.co"
    assert s.supabase_key == "test-key"
    assert s.phx_internal_token == "test-token"
    assert s.phx_company_min_match == 0.55
    assert s.phx_company_high_confidence == 0.82
    assert s.phx_company_ambiguity_margin == 0.10
    assert s.phx_employee_min_match == 0.55
    assert s.phx_employee_high_confidence == 0.86
    assert s.phx_employee_ambiguity_margin == 0.12
    assert s.phx_cache_ttl_seconds == 60
    assert s.phx_top_k_default == 5
    assert s.phx_log_payload is False
    assert s.phx_prewarm_enabled is True
```

- [ ] **Step 2: Run test — expect ImportError**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'phonetics_engine.config'`

- [ ] **Step 3: Write `src/phonetics_engine/__init__.py` (empty) and `src/phonetics_engine/config.py`**

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str = Field(...)
    supabase_key: str = Field(...)
    phx_internal_token: str = Field(...)

    phx_company_min_match: float = 0.55
    phx_company_high_confidence: float = 0.82
    phx_company_ambiguity_margin: float = 0.10

    phx_employee_min_match: float = 0.55
    phx_employee_high_confidence: float = 0.86
    phx_employee_ambiguity_margin: float = 0.12

    phx_cache_ttl_seconds: int = 60
    phx_top_k_default: int = 5
    phx_log_payload: bool = False
    phx_prewarm_enabled: bool = True
```

- [ ] **Step 4: Run test — PASS**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 1 passed.

- [ ] **Step 5: Create `.env.example`**

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=service-role-or-anon-key
PHX_INTERNAL_TOKEN=change-me

PHX_COMPANY_MIN_MATCH=0.55
PHX_COMPANY_HIGH_CONFIDENCE=0.82
PHX_COMPANY_AMBIGUITY_MARGIN=0.10
PHX_EMPLOYEE_MIN_MATCH=0.55
PHX_EMPLOYEE_HIGH_CONFIDENCE=0.86
PHX_EMPLOYEE_AMBIGUITY_MARGIN=0.12

PHX_CACHE_TTL_SECONDS=60
PHX_TOP_K_DEFAULT=5
PHX_LOG_PAYLOAD=0
PHX_PREWARM_ENABLED=1
```

- [ ] **Step 6: Commit**

```bash
git add src/phonetics_engine/__init__.py src/phonetics_engine/config.py tests/unit/test_config.py .env.example tests/__init__.py tests/unit/__init__.py
git commit -m "feat(config): pydantic-settings with all PHX_* env vars"
```

### Task 0.3: FastAPI app + GET /health

- [ ] **Step 1: Write failing test `tests/unit/test_health.py`**

```python
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


def test_health_returns_ok(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "t")
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 2: Run — fail (no module)**

Run: `uv run pytest tests/unit/test_health.py -v`
Expected: ImportError on `phonetics_engine.main`.

- [ ] **Step 3: Create `src/phonetics_engine/routes/__init__.py` (empty) and `src/phonetics_engine/routes/health.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 4: Create `src/phonetics_engine/main.py`**

```python
from fastapi import FastAPI

from phonetics_engine.routes import health


def create_app() -> FastAPI:
    app = FastAPI(title="phonetics-engine", version="0.1.0")
    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run test — PASS**

Run: `uv run pytest tests/unit/test_health.py -v`
Expected: 1 passed.

- [ ] **Step 6: Verify uvicorn runs**

Run: `uv run uvicorn phonetics_engine.main:app --port 8000 &`
Then: `curl -s http://localhost:8000/health`
Expected: `{"status":"ok"}`
Then: `kill %1`

- [ ] **Step 7: Commit**

```bash
git add src/phonetics_engine/main.py src/phonetics_engine/routes/
git add tests/unit/test_health.py
git commit -m "feat: FastAPI app factory + GET /health"
```

### Task 0.4: Dockerfile + render.yaml

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng libespeak-ng1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "phonetics_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `render.yaml`**

```yaml
services:
  - type: web
    name: phonetics-engine
    runtime: docker
    region: frankfurt
    plan: starter
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
      - key: PHX_INTERNAL_TOKEN
        sync: false
      - key: PHX_COMPANY_MIN_MATCH
        value: "0.55"
      - key: PHX_COMPANY_HIGH_CONFIDENCE
        value: "0.82"
      - key: PHX_COMPANY_AMBIGUITY_MARGIN
        value: "0.10"
      - key: PHX_EMPLOYEE_MIN_MATCH
        value: "0.55"
      - key: PHX_EMPLOYEE_HIGH_CONFIDENCE
        value: "0.86"
      - key: PHX_EMPLOYEE_AMBIGUITY_MARGIN
        value: "0.12"
      - key: PHX_CACHE_TTL_SECONDS
        value: "60"
      - key: PHX_TOP_K_DEFAULT
        value: "5"
      - key: PHX_LOG_PAYLOAD
        value: "0"
      - key: PHX_PREWARM_ENABLED
        value: "1"
```

- [ ] **Step 3: Create minimal `README.md`**

```markdown
# phonetics-engine

Multi-tenant phonetic matching service. See `GOAL.md` for the full spec and `docs/superpowers/plans/` for the implementation plan.

## Local dev

    uv sync --extra dev
    cp .env.example .env  # fill in real values
    uv run uvicorn phonetics_engine.main:app --reload

## Tests

    uv run pytest -v
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile render.yaml README.md
git commit -m "chore: Dockerfile + render.yaml + README"
```

---

## Phase 1 — Apply DB migration & verify

**Goal:** Migration `001_initial_schema.sql` applied to the dev/test Supabase project; service can read seed data.

**Files:** None (verification phase).

### Task 1.1: Apply migration via Supabase MCP

- [ ] **Step 1: Read the migration file**

Run: `cat migrations/001_initial_schema.sql | head -20`
Expected: header comment + `create table customers` line.

- [ ] **Step 2: Apply via Supabase MCP**

Use the `mcp__supabase__apply_migration` tool with `name="001_initial_schema"` and the full file body. (Or run it manually in the Supabase SQL editor if MCP is unavailable.)

- [ ] **Step 3: Verify tables exist**

Use `mcp__supabase__list_tables` and confirm: `customers`, `companies`, `employees`, `employee_company_roles` all present in the public schema.

- [ ] **Step 4: Verify seed data**

Use `mcp__supabase__execute_sql` with:

```sql
select c.first_name, c.last_name, c.infix, c.full_name, comp.canonical_name, ecr.phone
from employees c
join employee_company_roles ecr on ecr.employee_id = c.id
join companies comp on comp.id = ecr.company_id
where c.customer_id = 'xpots-dev'
order by c.first_name, comp.canonical_name;
```

Expected: 5 rows; "Steven de Vries" appears twice (taxameter, tmc) with phone `31621200435`; "Tristan van Doorn" once with infix `van`.

- [ ] **Step 5: Note down the project URL + a service-role key**

Use `mcp__supabase__get_project_url` and `mcp__supabase__get_publishable_keys` for the staging project. Store these in your local `.env` (do not commit).

---

## Phase 2 — Pydantic v2 models + enums

**Goal:** All wire-format types defined and validated. Phone is forbidden in `EmployeeCandidate`. Threshold-override is optional.

**Files:**
- Create: `src/phonetics_engine/enums.py`, `src/phonetics_engine/models.py`
- Test: `tests/unit/test_models.py`

### Task 2.1: Enums

- [ ] **Step 1: Write failing test `tests/unit/test_models.py`** (start with enums section)

```python
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
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: ImportError on `phonetics_engine.enums`.

- [ ] **Step 3: Write `src/phonetics_engine/enums.py`**

```python
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
    LAST_NAME = "last_name"
    LAST_NAME_WITH_INFIX = "last_name_with_infix"
    FULL_NAME = "full_name"
    ALIAS = "alias"
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/enums.py tests/unit/test_models.py
git commit -m "feat(enums): EntityType, Decision, MatchField"
```

### Task 2.2: Models — Thresholds, Scope, Candidate shapes, MatchRequest, Match, MatchResponse

- [ ] **Step 1: Add failing tests to `tests/unit/test_models.py`** (append to existing file)

```python
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
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: errors importing models from `phonetics_engine.models`.

- [ ] **Step 3: Write `src/phonetics_engine/models.py`**

```python
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
        return v


class Match(BaseModel):
    id: str
    display_name: str
    canonical_name: str
    score: float
    margin_to_next: float
    matched_field: MatchField | None = None
    matched_value: str | None = None


class MatchResponse(BaseModel):
    entity_type: EntityType
    decision: Decision
    applied_thresholds: Thresholds
    matches: list[Match]
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/models.py tests/unit/test_models.py
git commit -m "feat(models): MatchRequest/Response, Candidates, Thresholds; phone forbidden on employee"
```

---

## Phase 3 — Decision classification

**Goal:** Pure function that takes a normalized query, scored matches, and thresholds, and returns `(Decision, list[Match])`. Server-side, no I/O.

**Files:**
- Create: `src/phonetics_engine/normalize.py`, `src/phonetics_engine/decision.py`
- Test: `tests/unit/test_normalize.py`, `tests/unit/test_decision.py`

### Task 3.1: Canonicalize helper

- [ ] **Step 1: Failing test `tests/unit/test_normalize.py`**

```python
from phonetics_engine.normalize import canonicalize


def test_canonicalize_lowercases_and_strips():
    assert canonicalize("  Waysis  ") == "waysis"


def test_canonicalize_strips_diacritics():
    assert canonicalize("Café") == "cafe"
    assert canonicalize("Müller") == "muller"


def test_canonicalize_handles_empty():
    assert canonicalize("") == ""
    assert canonicalize("   ") == ""
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_normalize.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/phonetics_engine/normalize.py`**

```python
import unicodedata


def canonicalize(s: str) -> str:
    if not s:
        return ""
    decomposed = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower().strip()
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_normalize.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/normalize.py tests/unit/test_normalize.py
git commit -m "feat(normalize): NFKD-lower-strip canonicalize()"
```

### Task 3.2: Decision classifier

The classifier takes:
- `query` (raw, will be canonicalized inside)
- `scored` — list of `ScoredCandidate` (id, display_name, canonical_name, score, optional matched_field/matched_value) sorted desc by score
- `thresholds`
- `top_k`

It returns `(Decision, list[Match])` where each `Match` has `margin_to_next` filled in (last match has `margin_to_next = match.score`).

**Exact rule:** A candidate is "exact" if `canonicalize(query) == canonicalize(c.display_name)` OR `canonicalize(query) == c.canonical_name`. If exactly one candidate is exact → `EXACT`. If ≥2 are exact → `AMBIGUOUS`.

- [ ] **Step 1: Failing test `tests/unit/test_decision.py`**

```python
from dataclasses import dataclass

from phonetics_engine.enums import Decision, EntityType, MatchField
from phonetics_engine.decision import ScoredCandidate, classify
from phonetics_engine.models import Thresholds


COMPANY_THRESHOLDS = Thresholds(min_match=0.55, high_confidence=0.82, ambiguity_margin=0.10)


def _sc(id_, display, canonical, score, mf=None, mv=None):
    return ScoredCandidate(
        id=id_,
        display_name=display,
        canonical_name=canonical,
        score=score,
        matched_field=mf,
        matched_value=mv,
    )


def test_no_match_when_top_below_min():
    decision, matches = classify(
        query="ssspmlx",
        scored=[_sc("c1", "Waysis", "waysis", 0.20)],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.NO_MATCH
    assert matches == []


def test_exact_single_match():
    decision, matches = classify(
        query="Waysis",
        scored=[
            _sc("c1", "Waysis", "waysis", 0.99),
            _sc("c2", "Waste", "waste", 0.40),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.EXACT
    assert len(matches) == 1
    assert matches[0].id == "c1"
    assert matches[0].margin_to_next == matches[0].score  # only one returned


def test_exact_two_matches_becomes_ambiguous():
    decision, matches = classify(
        query="steven",
        scored=[
            _sc("e1", "Steven", "steven", 0.99),
            _sc("e2", "Steven", "steven", 0.99),
            _sc("e3", "Stefan", "stefan", 0.60),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 2
    assert {m.id for m in matches} == {"e1", "e2"}


def test_single_high_confidence():
    decision, matches = classify(
        query="wasteless",
        scored=[
            _sc("c1", "Waysis", "waysis", 0.91),
            _sc("c2", "Waste", "waste", 0.51),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.SINGLE_HIGH_CONFIDENCE
    assert len(matches) == 1
    assert matches[0].id == "c1"
    assert matches[0].margin_to_next == 0.40


def test_ambiguous_close_scores():
    decision, matches = classify(
        query="vries",
        scored=[
            _sc("e1", "Sanne de Vries", "sanne de vries", 0.80, MatchField.LAST_NAME, "Vries"),
            _sc("e2", "Bert Vries",      "bert vries",     0.78, MatchField.LAST_NAME, "Vries"),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 2


def test_top_k_truncates_ambiguous_matches():
    """All 10 candidates are within ambiguity_margin of the best — top_k should clip the returned list."""
    scored = [_sc(f"c{i}", f"Co{i}", f"co{i}", 0.90 - i * 0.01) for i in range(10)]
    decision, matches = classify(
        query="xxxxx",
        scored=scored,
        thresholds=COMPANY_THRESHOLDS,
        top_k=3,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 3


def test_margin_to_next_for_last_match():
    decision, matches = classify(
        query="xxxxx",
        scored=[
            _sc("c1", "A", "a", 0.95),
            _sc("c2", "B", "b", 0.50),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.SINGLE_HIGH_CONFIDENCE
    assert round(matches[0].margin_to_next, 4) == 0.45
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_decision.py -v`
Expected: ImportError on `phonetics_engine.decision`.

- [ ] **Step 3: Write `src/phonetics_engine/decision.py`**

```python
from dataclasses import dataclass

from phonetics_engine.enums import Decision, MatchField
from phonetics_engine.models import Match, Thresholds
from phonetics_engine.normalize import canonicalize


@dataclass
class ScoredCandidate:
    id: str
    display_name: str
    canonical_name: str
    score: float
    matched_field: MatchField | None = None
    matched_value: str | None = None


def _is_exact(query_canon: str, c: ScoredCandidate) -> bool:
    return query_canon == c.canonical_name or query_canon == canonicalize(c.display_name)


def _to_match(c: ScoredCandidate, margin: float) -> Match:
    return Match(
        id=c.id,
        display_name=c.display_name,
        canonical_name=c.canonical_name,
        score=round(c.score, 4),
        margin_to_next=round(margin, 4),
        matched_field=c.matched_field,
        matched_value=c.matched_value,
    )


def classify(
    query: str,
    scored: list[ScoredCandidate],
    thresholds: Thresholds,
    top_k: int,
) -> tuple[Decision, list[Match]]:
    if not scored:
        return Decision.NO_MATCH, []

    scored = sorted(scored, key=lambda c: c.score, reverse=True)
    query_canon = canonicalize(query)

    exact_matches = [c for c in scored if _is_exact(query_canon, c)]
    if len(exact_matches) == 1:
        return Decision.EXACT, [_to_match(exact_matches[0], exact_matches[0].score)]
    if len(exact_matches) >= 2:
        kept = exact_matches[:top_k]
        out = []
        for i, c in enumerate(kept):
            margin = c.score - kept[i + 1].score if i + 1 < len(kept) else c.score
            out.append(_to_match(c, margin))
        return Decision.AMBIGUOUS, out

    best = scored[0]
    if best.score < thresholds.min_match:
        return Decision.NO_MATCH, []

    runner_up_score = scored[1].score if len(scored) > 1 else 0.0
    margin = best.score - runner_up_score

    if margin < thresholds.ambiguity_margin:
        kept = [c for c in scored[:top_k] if c.score >= thresholds.min_match
                and (best.score - c.score) < thresholds.ambiguity_margin]
        out = []
        for i, c in enumerate(kept):
            m = c.score - kept[i + 1].score if i + 1 < len(kept) else c.score
            out.append(_to_match(c, m))
        return Decision.AMBIGUOUS, out

    if best.score >= thresholds.high_confidence:
        return Decision.SINGLE_HIGH_CONFIDENCE, [_to_match(best, margin)]

    # best.score in [min_match, high_confidence) and margin OK -> still no_match per spec
    return Decision.NO_MATCH, []
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_decision.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/decision.py tests/unit/test_decision.py
git commit -m "feat(decision): server-side classify() for all 5 decisions + tie-breaker"
```

---

## Phase 4 — Phonetic core (port from Bonove)

**Goal:** `phonemize_name`, `phonemize_batch`, `_phonemes_to_vector`, and `PhoneticIndex.search` available locally; identical behavior to Bonove's `phonetics.py`.

**Files:**
- Create: `src/phonetics_engine/phonetics.py`
- Test: `tests/unit/test_phonetics.py`

### Task 4.1: Port phonetics.py

- [ ] **Step 1: Failing test `tests/unit/test_phonetics.py`**

```python
import pytest

from phonetics_engine.phonetics import (
    PhoneticIndex,
    phonemize_batch,
    phonemize_name,
)


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


def test_phonemize_name_returns_phonemes():
    p = phonemize_name("Steven")
    assert p
    assert " " in p  # spaces between phonemes


def test_phonemize_batch_preserves_length():
    out = phonemize_batch(["Steven", "", "Wasteless"])
    assert len(out) == 3
    assert out[1] == ""


def test_index_search_finds_close_match():
    idx = PhoneticIndex(["Steven", "Stefan", "Marie", "Wasteless"])
    results = idx.search("Steeve", top_k=3)
    assert results
    assert results[0]["name"] in {"Steven", "Stefan"}
    assert 0.0 <= results[0]["score"] <= 1.0


def test_index_empty_returns_empty_search():
    idx = PhoneticIndex([])
    assert idx.search("anything") == []
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_phonetics.py -v`
Expected: ImportError or skipped.

- [ ] **Step 3: Write `src/phonetics_engine/phonetics.py`**

```python
import numpy as np
import faiss
from phonemizer import phonemize
from phonemizer.separator import Separator

PHONEMIZER_LANGUAGE = "nl"
PHONEMIZER_BACKEND = "espeak"
SIMILARITY_THRESHOLD = 0.0  # we filter at decision-layer with thresholds; keep raw here

_SEPARATOR = Separator(phone=" ", word="", syllable="")


def phonemize_name(name: str) -> str:
    if not name or not name.strip():
        return ""
    result = phonemize(
        name.strip(),
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )
    return result.strip()


def phonemize_batch(names: list[str]) -> list[str]:
    cleaned = [n.strip() if n else "" for n in names]
    non_empty = [n for n in cleaned if n]

    if not non_empty:
        return [""] * len(names)

    results = phonemize(
        non_empty,
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )

    out: list[str] = []
    rit = iter(results if isinstance(results, list) else [results])
    for n in cleaned:
        if n:
            out.append(next(rit).strip())
        else:
            out.append("")
    return out


def _phonemes_to_vector(phonemes: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    chars = phonemes.replace(" ", "")
    if not chars:
        return vec
    for n in (2, 3):
        for i in range(len(chars) - n + 1):
            ngram = chars[i : i + n]
            h = hash(ngram) % dim
            vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class PhoneticIndex:
    def __init__(self, names: list[str]):
        self._names = list(names)
        self._phonemes: list[str] = []
        self._index: faiss.Index | None = None

        if not names:
            return

        self._phonemes = phonemize_batch(names)
        dim = 128
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in self._phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._names)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._index or self.size == 0:
            return []
        qp = phonemize_name(query)
        if not qp:
            return []
        qv = _phonemes_to_vector(qp).reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self._index.search(qv, k)
        results: list[dict] = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            clamped = float(max(0.0, min(1.0, s)))
            if clamped >= SIMILARITY_THRESHOLD:
                results.append(
                    {"name": self._names[i], "score": round(clamped, 4), "phonemes": self._phonemes[i]}
                )
        return results
```

- [ ] **Step 4: Run — PASS** (requires espeak-ng on the system; install with `brew install espeak-ng` on macOS)

Run: `uv run pytest tests/unit/test_phonetics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/phonetics.py tests/unit/test_phonetics.py
git commit -m "feat(phonetics): phonemize + FAISS PhoneticIndex (Bonove port)"
```

---

## Phase 5 — Loader (Supabase REST)

**Goal:** Async functions that pull `companies` and `employees` (with their roles) for a given `customer_id` from Supabase via REST API. Mockable in tests via `respx`.

**Files:**
- Create: `src/phonetics_engine/loader.py`
- Test: `tests/unit/test_loader.py`

### Task 5.1: Loader records + queries

- [ ] **Step 1: Failing test `tests/unit/test_loader.py`**

```python
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
        {"id": "c2", "display_name": "TaxiCentrale Maassluis", "canonical_name": "taxicentrale maassluis", "aliases": ["TCM"]},
    ]))

    out = await fetch_companies(settings, customer_id="1000435")
    assert len(out) == 2
    assert isinstance(out[0], CompanyRecord)
    assert out[0].canonical_name == "waysis"
    assert out[1].aliases == ["TCM"]


@respx.mock
@pytest.mark.asyncio
async def test_fetch_employees_joins_roles(settings):
    respx.get(
        "https://test.supabase.co/rest/v1/employees",
        params={
            "customer_id": "eq.1000435",
            "select": "id,first_name,infix,last_name,full_name,employee_company_roles(company_id)",
        },
    ).mock(return_value=httpx.Response(200, json=[
        {
            "id": "e1",
            "first_name": "Sanne",
            "infix": "de",
            "last_name": "Vries",
            "full_name": "Sanne de Vries",
            "employee_company_roles": [{"company_id": "c1"}, {"company_id": "c2"}],
        },
    ]))
    out = await fetch_employees(settings, customer_id="1000435")
    assert len(out) == 1
    assert out[0].full_name == "Sanne de Vries"
    assert set(out[0].company_ids) == {"c1", "c2"}


@respx.mock
@pytest.mark.asyncio
async def test_fetch_employees_filters_by_company_id(settings):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "A", "infix": None, "last_name": "X", "full_name": "A X",
             "employee_company_roles": [{"company_id": "c1"}]},
            {"id": "e2", "first_name": "B", "infix": None, "last_name": "Y", "full_name": "B Y",
             "employee_company_roles": [{"company_id": "c2"}]},
        ])
    )
    out = await fetch_employees(settings, customer_id="1000435", company_id="c1")
    assert {e.id for e in out} == {"e1"}


@respx.mock
@pytest.mark.asyncio
async def test_fetch_companies_propagates_error(settings):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(500, text="boom")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await fetch_companies(settings, customer_id="1000435")
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_loader.py -v`
Expected: ImportError on `phonetics_engine.loader`.

- [ ] **Step 3: Write `src/phonetics_engine/loader.py`**

```python
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
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_loader.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/loader.py tests/unit/test_loader.py
git commit -m "feat(loader): async Supabase REST queries for companies/employees"
```

---

## Phase 6 — Matcher (per-tenant index build + multi-vector employees)

**Goal:** Convert lists of `CompanyRecord`/`EmployeeRecord` into FAISS-backed scoring. Employees get **two** vectors per record (last_name and infix+last_name); company matching is on display_name + canonical_name + aliases.

**Files:**
- Create: `src/phonetics_engine/matcher.py`
- Test: `tests/unit/test_matcher.py`

### Task 6.1: TenantIndex builder

The tenant-index holds a flat list of `(record_id, matched_field, matched_value, vector)` tuples and a single FAISS `IndexFlatIP` over them. `search(query, top_k)` returns deduplicated `ScoredCandidate`s — for one record we keep the highest-scoring matched_field.

- [ ] **Step 1: Failing test `tests/unit/test_matcher.py`**

```python
import pytest

from phonetics_engine.enums import MatchField
from phonetics_engine.loader import CompanyRecord, EmployeeRecord
from phonetics_engine.matcher import build_company_index, build_employee_index


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


def _company(id_, display, canonical, aliases=None):
    return CompanyRecord(id=id_, display_name=display, canonical_name=canonical, aliases=aliases or [])


def _employee(id_, first, infix, last):
    full = f"{first} {infix + ' ' if infix else ''}{last}".strip()
    return EmployeeRecord(id=id_, first_name=first, infix=infix, last_name=last, full_name=full, company_ids=[])


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
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_matcher.py -v`
Expected: ImportError on `phonetics_engine.matcher`.

- [ ] **Step 3: Write `src/phonetics_engine/matcher.py`**

```python
from dataclasses import dataclass

import faiss
import numpy as np

from phonetics_engine.decision import ScoredCandidate
from phonetics_engine.enums import MatchField
from phonetics_engine.loader import CompanyRecord, EmployeeRecord
from phonetics_engine.phonetics import _phonemes_to_vector, phonemize_batch, phonemize_name


@dataclass
class _Entry:
    record_id: str
    display_name: str
    canonical_name: str
    matched_field: MatchField
    matched_value: str


class TenantIndex:
    def __init__(self, entries: list[_Entry], dim: int = 128):
        self._entries = entries
        self._dim = dim
        self._index: faiss.Index | None = None

        if not entries:
            return

        phonemes = phonemize_batch([e.matched_value for e in entries])
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    def search(self, query: str, top_k: int) -> list[ScoredCandidate]:
        if not self._index or not self._entries:
            return []

        qp = phonemize_name(query)
        if not qp:
            return []
        qv = _phonemes_to_vector(qp, self._dim).reshape(1, -1)

        # Pull more raw matches than top_k since dedup may reduce the result count.
        k_raw = min(len(self._entries), max(top_k * 4, 16))
        scores, indices = self._index.search(qv, k_raw)

        best: dict[str, ScoredCandidate] = {}
        for s, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            entry = self._entries[i]
            clamped = float(max(0.0, min(1.0, s)))
            existing = best.get(entry.record_id)
            if existing is None or clamped > existing.score:
                best[entry.record_id] = ScoredCandidate(
                    id=entry.record_id,
                    display_name=entry.display_name,
                    canonical_name=entry.canonical_name,
                    score=clamped,
                    matched_field=entry.matched_field,
                    matched_value=entry.matched_value,
                )

        out = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return out[:top_k]


def build_company_index(records: list[CompanyRecord], *, match_fields: list[MatchField]) -> TenantIndex:
    entries: list[_Entry] = []
    for r in records:
        if MatchField.DISPLAY_NAME in match_fields:
            entries.append(_Entry(r.id, r.display_name, r.canonical_name, MatchField.DISPLAY_NAME, r.display_name))
        if MatchField.CANONICAL_NAME in match_fields:
            entries.append(_Entry(r.id, r.display_name, r.canonical_name, MatchField.CANONICAL_NAME, r.canonical_name))
        if MatchField.ALIAS in match_fields:
            for alias in r.aliases:
                entries.append(_Entry(r.id, r.display_name, r.canonical_name, MatchField.ALIAS, alias))
    return TenantIndex(entries)


def build_employee_index(records: list[EmployeeRecord], *, match_fields: list[MatchField]) -> TenantIndex:
    entries: list[_Entry] = []
    for r in records:
        display = r.full_name
        canonical = r.full_name.lower()
        if MatchField.LAST_NAME in match_fields:
            entries.append(_Entry(r.id, display, canonical, MatchField.LAST_NAME, r.last_name))
            if r.infix and r.infix.strip():
                value = f"{r.infix.strip()} {r.last_name}"
                entries.append(_Entry(r.id, display, canonical, MatchField.LAST_NAME_WITH_INFIX, value))
        if MatchField.FULL_NAME in match_fields:
            entries.append(_Entry(r.id, display, canonical, MatchField.FULL_NAME, r.full_name))
    return TenantIndex(entries)
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_matcher.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/matcher.py tests/unit/test_matcher.py
git commit -m "feat(matcher): TenantIndex with multi-field entries + dedup"
```

---

## Phase 7 — IndexCache (TTL + concurrency lock)

**Goal:** Per-key cache of `TenantIndex` with TTL and an `asyncio.Lock` per cache-key so parallel cache-misses build the index only once.

**Files:**
- Create: `src/phonetics_engine/index_cache.py`
- Test: `tests/unit/test_index_cache.py`

### Task 7.1: TTL cache with async lock

- [ ] **Step 1: Failing test `tests/unit/test_index_cache.py`**

```python
import asyncio
import time

import pytest

from phonetics_engine.index_cache import IndexCache


@pytest.mark.asyncio
async def test_cache_miss_calls_loader_once():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return f"index-{calls}"

    cache = IndexCache(ttl_seconds=60.0)
    out = await cache.get_or_build("k1", loader)
    assert out == "index-1"
    out2 = await cache.get_or_build("k1", loader)
    assert out2 == "index-1"  # cached
    assert calls == 1


@pytest.mark.asyncio
async def test_parallel_misses_call_loader_once():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "the-index"

    cache = IndexCache(ttl_seconds=60.0)
    results = await asyncio.gather(*(cache.get_or_build("k1", loader) for _ in range(5)))
    assert all(r == "the-index" for r in results)
    assert calls == 1


@pytest.mark.asyncio
async def test_ttl_expiry_triggers_rebuild():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return f"v{calls}"

    cache = IndexCache(ttl_seconds=0.05)
    a = await cache.get_or_build("k", loader)
    assert a == "v1"
    await asyncio.sleep(0.07)
    b = await cache.get_or_build("k", loader)
    assert b == "v2"
    assert calls == 2


@pytest.mark.asyncio
async def test_invalidate_drops_entry():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return f"v{calls}"

    cache = IndexCache(ttl_seconds=60.0)
    await cache.get_or_build("k", loader)
    cache.invalidate("k")
    await cache.get_or_build("k", loader)
    assert calls == 2


@pytest.mark.asyncio
async def test_invalidate_prefix():
    calls = 0

    async def loader():
        nonlocal calls
        calls += 1
        return calls

    cache = IndexCache(ttl_seconds=60.0)
    await cache.get_or_build(("cust", "company"), loader)
    await cache.get_or_build(("cust", "employee"), loader)
    cache.invalidate_prefix(("cust",))
    await cache.get_or_build(("cust", "company"), loader)
    assert calls == 3
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_index_cache.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `src/phonetics_engine/index_cache.py`**

```python
import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Generic, Hashable, TypeVar

T = TypeVar("T")


class IndexCache(Generic[T]):
    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._values: dict[Hashable, tuple[float, T]] = {}
        self._locks: dict[Hashable, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _is_fresh(self, expires_at: float) -> bool:
        return time.monotonic() < expires_at

    async def _key_lock(self, key: Hashable) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def get_or_build(self, key: Hashable, builder: Callable[[], Awaitable[T]]) -> T:
        entry = self._values.get(key)
        if entry is not None and self._is_fresh(entry[0]):
            return entry[1]

        lock = await self._key_lock(key)
        async with lock:
            entry = self._values.get(key)
            if entry is not None and self._is_fresh(entry[0]):
                return entry[1]
            built = await builder()
            self._values[key] = (time.monotonic() + self._ttl, built)
            return built

    def invalidate(self, key: Hashable) -> None:
        self._values.pop(key, None)

    def invalidate_prefix(self, prefix: tuple[Any, ...]) -> None:
        plen = len(prefix)
        to_drop = [
            k for k in list(self._values.keys())
            if isinstance(k, tuple) and len(k) >= plen and k[:plen] == prefix
        ]
        for k in to_drop:
            self._values.pop(k, None)
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_index_cache.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/index_cache.py tests/unit/test_index_cache.py
git commit -m "feat(index_cache): TTL + per-key asyncio.Lock + prefix-invalidate"
```

---

## Phase 8 — Auth middleware

**Goal:** FastAPI dependency that validates `X-Internal-Token` header; returns 401 if missing or wrong.

**Files:**
- Create: `src/phonetics_engine/auth.py`
- Test: `tests/unit/test_auth.py`

### Task 8.1: Header dependency

- [ ] **Step 1: Failing test `tests/unit/test_auth.py`**

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from phonetics_engine.auth import require_internal_token
from phonetics_engine.config import Settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret-token")

    app = FastAPI()

    @app.get("/protected", dependencies=[require_internal_token()])
    def protected():
        return {"ok": True}

    return TestClient(app)


def test_missing_header_returns_401(client):
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_token_returns_401(client):
    r = client.get("/protected", headers={"X-Internal-Token": "nope"})
    assert r.status_code == 401


def test_correct_token_passes(client):
    r = client.get("/protected", headers={"X-Internal-Token": "secret-token"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/unit/test_auth.py -v`
Expected: ImportError on `phonetics_engine.auth`.

- [ ] **Step 3: Write `src/phonetics_engine/auth.py`**

```python
from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from phonetics_engine.config import Settings


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


def _verify(x_internal_token: str | None = Header(default=None)) -> None:
    expected = _settings().phx_internal_token
    if not x_internal_token or x_internal_token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")


def require_internal_token() -> Depends:
    return Depends(_verify)
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/unit/test_auth.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/auth.py tests/unit/test_auth.py
git commit -m "feat(auth): X-Internal-Token FastAPI dependency"
```

---

## Phase 9 — POST /v1/match (happy path)

**Goal:** Endpoint that ties everything together for the standard "Carla calls with `customer_id`, no overrides" case. Status-policy and override-handling come in Phase 10.

**Files:**
- Create: `src/phonetics_engine/routes/match.py`
- Modify: `src/phonetics_engine/main.py:1-15` (mount router)
- Test: `tests/integration/test_match_endpoint.py`

### Task 9.1: Endpoint wiring

The endpoint pulls the cached `TenantIndex` for `(customer_id, entity_type [, company_id])`. Cache key:
- Company: `(customer_id, "company")`
- Employee with scope: `(customer_id, "employee", company_id)`
- Employee without scope: `(customer_id, "employee")`

Default `match_fields`:
- Company: `[DISPLAY_NAME, CANONICAL_NAME, ALIAS]`
- Employee: `[LAST_NAME, FULL_NAME]`

Default thresholds: from `Settings` per `entity_type`.

- [ ] **Step 1: Failing test `tests/integration/test_match_endpoint.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_match_company_single_high_confidence(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
            {"id": "c2", "display_name": "Wasteless", "canonical_name": "wasteless", "aliases": []},
        ])
    )

    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "waysis", "entity_type": "company", "customer_id": "xpots-dev"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["entity_type"] == "company"
    assert body["decision"] in ("exact", "single_high_confidence")
    assert body["matches"][0]["id"] == "c1"
    assert "phone" not in body["matches"][0]


@respx.mock
def test_match_employee_with_scope_filters(client):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": "de", "last_name": "Vries",
             "full_name": "Sanne de Vries", "employee_company_roles": [{"company_id": "c1"}]},
            {"id": "e2", "first_name": "Bert", "infix": None, "last_name": "Jansen",
             "full_name": "Bert Jansen", "employee_company_roles": [{"company_id": "c2"}]},
        ])
    )

    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Vries",
            "entity_type": "employee",
            "customer_id": "xpots-dev",
            "scope": {"company_id": "c1"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["matches"][0]["id"] == "e1"


def test_match_requires_token(client):
    r = client.post("/v1/match", json={"query": "x", "entity_type": "company", "customer_id": "x"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_match_endpoint.py -v`
Expected: ImportError or 404 on `/v1/match`.

- [ ] **Step 3: Create `tests/integration/__init__.py` (empty) and write `src/phonetics_engine/routes/match.py`**

```python
from functools import lru_cache

from fastapi import APIRouter, Request

from phonetics_engine.auth import require_internal_token
from phonetics_engine.config import Settings
from phonetics_engine.decision import classify
from phonetics_engine.enums import EntityType, MatchField
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.loader import fetch_companies, fetch_employees
from phonetics_engine.matcher import TenantIndex, build_company_index, build_employee_index
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
    return _company_thresholds(s) if req.entity_type == EntityType.COMPANY else _employee_thresholds(s)


def _resolve_fields(req: MatchRequest) -> list[MatchField]:
    if req.match_fields:
        return req.match_fields
    return _DEFAULT_FIELDS_COMPANY if req.entity_type == EntityType.COMPANY else _DEFAULT_FIELDS_EMPLOYEE


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
```

- [ ] **Step 4: Update `src/phonetics_engine/main.py` to mount the router and an `IndexCache` on `app.state`**

```python
from fastapi import FastAPI

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.routes import health, match


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="phonetics-engine", version="0.1.0")
    app.state.settings = settings
    app.state.index_cache = IndexCache(ttl_seconds=float(settings.phx_cache_ttl_seconds))
    app.include_router(health.router)
    app.include_router(match.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run tests — PASS**

Run: `uv run pytest tests/integration/test_match_endpoint.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/phonetics_engine/routes/match.py src/phonetics_engine/main.py tests/integration/__init__.py tests/integration/test_match_endpoint.py
git commit -m "feat(match): POST /v1/match happy path with cache + decision-classify"
```

---

## Phase 10 — POST /v1/match: overrides + HTTP-status policy

**Goal:** Add candidates override, threshold override (already partially wired via `_resolve_thresholds`), and **enforce always-200-except-401/422** by catching everything in a try/except and returning `decision: "service_error"` on unexpected failure. Add tests.

**Files:**
- Modify: `src/phonetics_engine/routes/match.py` (entire file rewrite)
- Test: `tests/integration/test_status_policy.py`, `tests/integration/test_threshold_override.py`, `tests/integration/test_phone_strip.py`, `tests/integration/test_tenant_isolation.py`, `tests/integration/test_concurrency_lock.py`, `tests/integration/test_tussenvoegsel_matching.py`

### Task 10.1: Status policy + service_error fallback

- [ ] **Step 1: Failing test `tests/integration/test_status_policy.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_supabase_5xx_becomes_200_service_error(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(500, text="boom")
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "company", "customer_id": "any"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "service_error"
    assert body["matches"] == []


@respx.mock
def test_empty_tenant_returns_200_no_match(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "company", "customer_id": "empty-tenant"},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "no_match"


def test_invalid_entity_type_returns_422(client):
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "x", "entity_type": "vehicle", "customer_id": "x"},
    )
    assert r.status_code == 422


def test_missing_token_returns_401(client):
    r = client.post(
        "/v1/match",
        json={"query": "x", "entity_type": "company", "customer_id": "x"},
    )
    assert r.status_code == 401
```

- [ ] **Step 2: Run — fail (5xx test will fail because the unhandled exception bubbles)**

Run: `uv run pytest tests/integration/test_status_policy.py -v`
Expected: 5xx case fails (returns 500), the 422/401 cases pass.

- [ ] **Step 3: Modify `src/phonetics_engine/routes/match.py` — wrap business logic in try/except**

Replace the `match` handler body:

```python
import logging

logger = logging.getLogger(__name__)


@router.post("/v1/match", response_model=MatchResponse, dependencies=[require_internal_token()])
async def match(req: MatchRequest, request: Request) -> MatchResponse:
    s = _settings()
    cache: IndexCache = request.app.state.index_cache
    thresholds = _resolve_thresholds(req, s)

    try:
        if req.candidates is not None:
            scored = _search_candidates_override(req, s)
        else:
            async def builder() -> TenantIndex:
                if req.entity_type == EntityType.COMPANY:
                    return await _build_company_tenant_index(req, s)
                return await _build_employee_tenant_index(req, s)

            index = await cache.get_or_build(_cache_key(req), builder)
            scored = index.search(req.query, top_k=req.top_k)

        decision, matches = classify(req.query, scored, thresholds, req.top_k)
    except Exception:
        logger.exception("match_service_error customer_id=%s entity_type=%s", req.customer_id, req.entity_type.value)
        return MatchResponse(
            entity_type=req.entity_type,
            decision=Decision.SERVICE_ERROR,
            applied_thresholds=thresholds,
            matches=[],
        )

    return MatchResponse(
        entity_type=req.entity_type,
        decision=decision,
        applied_thresholds=thresholds,
        matches=matches,
    )
```

Also import `Decision` at the top of `match.py`:

```python
from phonetics_engine.enums import Decision, EntityType, MatchField
```

And add the candidates-override helper (next task fills the body; for now stub):

```python
def _search_candidates_override(req: MatchRequest, s: Settings):
    raise NotImplementedError("filled in next task")
```

- [ ] **Step 4: Run — PASS for status-policy tests**

Run: `uv run pytest tests/integration/test_status_policy.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/routes/match.py tests/integration/test_status_policy.py
git commit -m "feat(match): always-200-except-401/422 status policy with service_error fallback"
```

### Task 10.2: Candidates override

- [ ] **Step 1: Failing test (extend `tests/integration/test_match_endpoint.py`)**

```python
def test_candidates_override_skips_db(client):
    # No respx stubs registered — if the loader is called, this would fail.
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Waysis",
            "entity_type": "company",
            "customer_id": "any",
            "candidates": [
                {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis"},
                {"id": "c2", "display_name": "Wasteless", "canonical_name": "wasteless"},
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["matches"][0]["id"] == "c1"
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_match_endpoint.py::test_candidates_override_skips_db -v`
Expected: NotImplementedError → service_error.

- [ ] **Step 3: Implement `_search_candidates_override` in `src/phonetics_engine/routes/match.py`**

Add an import at the top:

```python
from phonetics_engine.loader import CompanyRecord, EmployeeRecord
from phonetics_engine.models import CompanyCandidate, EmployeeCandidate
```

Replace the stub:

```python
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
```

- [ ] **Step 4: Run — PASS**

Run: `uv run pytest tests/integration/test_match_endpoint.py -v`
Expected: 4 passed (3 prior + new override test).

- [ ] **Step 5: Commit**

```bash
git add src/phonetics_engine/routes/match.py tests/integration/test_match_endpoint.py
git commit -m "feat(match): candidates override (skip DB, build ad-hoc index)"
```

### Task 10.3: Threshold override + applied_thresholds

- [ ] **Step 1: Failing test `tests/integration/test_threshold_override.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_request_thresholds_override_env_defaults(client):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "wasteless",
            "entity_type": "company",
            "customer_id": "x",
            "thresholds": {"min_match": 0.10, "high_confidence": 0.20, "ambiguity_margin": 0.01},
        },
    )
    body = r.json()
    assert body["applied_thresholds"]["min_match"] == 0.10
    assert body["applied_thresholds"]["high_confidence"] == 0.20
    assert body["applied_thresholds"]["ambiguity_margin"] == 0.01


@respx.mock
def test_default_thresholds_use_env_for_employee(client):
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": None, "last_name": "Vries",
             "full_name": "Sanne Vries", "employee_company_roles": []},
        ])
    )
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "Vries", "entity_type": "employee", "customer_id": "x"},
    )
    body = r.json()
    assert body["applied_thresholds"]["min_match"] == 0.55
    assert body["applied_thresholds"]["high_confidence"] == 0.86  # employee default
    assert body["applied_thresholds"]["ambiguity_margin"] == 0.12
```

- [ ] **Step 2: Run — PASS** (already implemented in Task 9 / 10.1 — `_resolve_thresholds` honors `req.thresholds`)

Run: `uv run pytest tests/integration/test_threshold_override.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_threshold_override.py
git commit -m "test(match): threshold override + applied_thresholds in response"
```

### Task 10.4: Phone-strip regression test

- [ ] **Step 1: `tests/integration/test_phone_strip.py`**

```python
import pytest
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


def test_employee_candidate_with_phone_returns_422(client):
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={
            "query": "Vries",
            "entity_type": "employee",
            "customer_id": "x",
            "candidates": [
                {"id": "e1", "first_name": "Sanne", "last_name": "Vries", "phone": "31600000000"},
            ],
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert "phone" in str(body).lower()
```

- [ ] **Step 2: Run — PASS** (extra="forbid" on `EmployeeCandidate` already enforces this; test confirms)

Run: `uv run pytest tests/integration/test_phone_strip.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_phone_strip.py
git commit -m "test(match): phone in EmployeeCandidate -> 422"
```

### Task 10.5: Tenant-isolation regression

- [ ] **Step 1: `tests/integration/test_tenant_isolation.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_customer_id_filter_is_passed_to_supabase(client):
    route_a = respx.get(
        "https://test.supabase.co/rest/v1/companies",
        params={"customer_id": "eq.tenant-a", "select": "id,display_name,canonical_name,aliases"},
    ).mock(return_value=httpx.Response(200, json=[
        {"id": "ca", "display_name": "Alpha", "canonical_name": "alpha", "aliases": []},
    ]))

    route_b = respx.get(
        "https://test.supabase.co/rest/v1/companies",
        params={"customer_id": "eq.tenant-b", "select": "id,display_name,canonical_name,aliases"},
    ).mock(return_value=httpx.Response(200, json=[
        {"id": "cb", "display_name": "Beta", "canonical_name": "beta", "aliases": []},
    ]))

    r_a = client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                     json={"query": "alpha", "entity_type": "company", "customer_id": "tenant-a"})
    r_b = client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                     json={"query": "beta", "entity_type": "company", "customer_id": "tenant-b"})

    assert route_a.called and route_b.called
    assert r_a.json()["matches"][0]["id"] == "ca"
    assert r_b.json()["matches"][0]["id"] == "cb"
```

- [ ] **Step 2: Run — PASS**

Run: `uv run pytest tests/integration/test_tenant_isolation.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_tenant_isolation.py
git commit -m "test(match): tenant isolation — customer_id passed to Supabase"
```

### Task 10.6: Concurrency-lock regression

- [ ] **Step 1: `tests/integration/test_concurrency_lock.py`**

```python
import asyncio
import time

import httpx
import pytest
import respx

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return create_app()


@respx.mock
@pytest.mark.asyncio
async def test_parallel_misses_call_supabase_once(app):
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)
        return httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])

    respx.get("https://test.supabase.co/rest/v1/companies").mock(side_effect=handler)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        results = await asyncio.gather(*(
            client.post(
                "/v1/match",
                headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "tenant-x"},
            )
            for _ in range(5)
        ))
    assert all(r.status_code == 200 for r in results)
    assert call_count == 1
```

- [ ] **Step 2: Run — PASS** (the IndexCache lock guarantees this)

Run: `uv run pytest tests/integration/test_concurrency_lock.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_concurrency_lock.py
git commit -m "test(match): concurrency-lock — 5 parallel misses build index 1x"
```

### Task 10.7: Tussenvoegsel regression

- [ ] **Step 1: `tests/integration/test_tussenvoegsel_matching.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


def _stub_supabase():
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "Sanne", "infix": "de", "last_name": "Vries",
             "full_name": "Sanne de Vries", "employee_company_roles": []},
        ])
    )


@respx.mock
def test_query_last_name_only(client):
    _stub_supabase()
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "Vries", "entity_type": "employee", "customer_id": "t"},
    )
    body = r.json()
    assert body["matches"][0]["id"] == "e1"
    assert body["matches"][0]["matched_field"] == "last_name"
    assert body["matches"][0]["matched_value"] == "Vries"


@respx.mock
def test_query_with_infix(client):
    _stub_supabase()
    r = client.post(
        "/v1/match",
        headers={"X-Internal-Token": "secret"},
        json={"query": "de Vries", "entity_type": "employee", "customer_id": "t"},
    )
    body = r.json()
    assert body["matches"][0]["id"] == "e1"
    assert body["matches"][0]["matched_field"] == "last_name_with_infix"
    assert body["matches"][0]["matched_value"] == "de Vries"
```

- [ ] **Step 2: Run — PASS**

Run: `uv run pytest tests/integration/test_tussenvoegsel_matching.py -v`
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_tussenvoegsel_matching.py
git commit -m "test(match): infix matching returns correct matched_field/matched_value"
```

---

## Phase 11 — POST /v1/reload

**Goal:** Authenticated cache flush. Body: `{"customer_id": "...", "entity_type": "company" | "employee" | null}`. `entity_type` null = customer-wide.

**Files:**
- Create: `src/phonetics_engine/routes/reload.py`
- Modify: `src/phonetics_engine/main.py:1-15` (mount router)
- Test: `tests/integration/test_reload_endpoint.py`

### Task 11.1: Reload endpoint

- [ ] **Step 1: Failing test `tests/integration/test_reload_endpoint.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_reload_invalidates_specific_entity(client):
    call_count = 0

    def handler(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])

    respx.get("https://test.supabase.co/rest/v1/companies").mock(side_effect=handler)

    # First call -> miss -> Supabase hit (call 1)
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    # Second call -> hit -> no Supabase
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    assert call_count == 1

    # Reload company-only
    r = client.post("/v1/reload",
                    headers={"X-Internal-Token": "secret"},
                    json={"customer_id": "t", "entity_type": "company"})
    assert r.status_code == 200
    assert r.json() == {"flushed": True, "customer_id": "t", "entity_type": "company"}

    # Third call -> miss again -> Supabase hit (call 2)
    client.post("/v1/match", headers={"X-Internal-Token": "secret"},
                json={"query": "waysis", "entity_type": "company", "customer_id": "t"})
    assert call_count == 2


def test_reload_requires_token(client):
    r = client.post("/v1/reload", json={"customer_id": "x"})
    assert r.status_code == 401


def test_reload_customer_wide_when_entity_type_null(client):
    r = client.post("/v1/reload", headers={"X-Internal-Token": "secret"},
                    json={"customer_id": "t"})
    assert r.status_code == 200
    assert r.json() == {"flushed": True, "customer_id": "t", "entity_type": None}
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_reload_endpoint.py -v`
Expected: 404 on `/v1/reload`.

- [ ] **Step 3: Write `src/phonetics_engine/routes/reload.py`**

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict

from phonetics_engine.auth import require_internal_token
from phonetics_engine.enums import EntityType
from phonetics_engine.index_cache import IndexCache


router = APIRouter()


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str
    entity_type: EntityType | None = None


class ReloadResponse(BaseModel):
    flushed: bool
    customer_id: str
    entity_type: EntityType | None


@router.post("/v1/reload", response_model=ReloadResponse, dependencies=[require_internal_token()])
async def reload(req: ReloadRequest, request: Request) -> ReloadResponse:
    cache: IndexCache = request.app.state.index_cache
    if req.entity_type is None:
        cache.invalidate_prefix((req.customer_id,))
    else:
        cache.invalidate_prefix((req.customer_id, req.entity_type.value))
    return ReloadResponse(flushed=True, customer_id=req.customer_id, entity_type=req.entity_type)
```

- [ ] **Step 4: Mount in `src/phonetics_engine/main.py`**

```python
from phonetics_engine.routes import health, match, reload as reload_route

# inside create_app():
app.include_router(reload_route.router)
```

- [ ] **Step 5: Run — PASS**

Run: `uv run pytest tests/integration/test_reload_endpoint.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/phonetics_engine/routes/reload.py src/phonetics_engine/main.py tests/integration/test_reload_endpoint.py
git commit -m "feat(reload): POST /v1/reload with optional entity_type"
```

---

## Phase 12 — Pre-warm at startup

**Goal:** When `PHX_PREWARM_ENABLED=1`, on startup spawn a background task that lists known `customer_id`s from Supabase, then builds company + employee indexes for each. Service is healthy immediately; pre-warm runs concurrently.

**Files:**
- Create: `src/phonetics_engine/prewarm.py`
- Modify: `src/phonetics_engine/main.py` (lifespan hook)
- Test: `tests/integration/test_prewarm.py`

### Task 12.1: Prewarm task

- [ ] **Step 1: Failing test `tests/integration/test_prewarm.py`**

```python
import asyncio

import httpx
import pytest
import respx

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.prewarm import prewarm_all


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "t")
    return Settings()


@respx.mock
@pytest.mark.asyncio
async def test_prewarm_lists_tenants_and_builds_indexes(settings):
    respx.get("https://test.supabase.co/rest/v1/customers").mock(
        return_value=httpx.Response(200, json=[{"id": "tenant-a"}, {"id": "tenant-b"}])
    )
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "X", "canonical_name": "x", "aliases": []},
        ])
    )
    respx.get("https://test.supabase.co/rest/v1/employees").mock(
        return_value=httpx.Response(200, json=[
            {"id": "e1", "first_name": "A", "infix": None, "last_name": "Z",
             "full_name": "A Z", "employee_company_roles": []},
        ])
    )

    cache = IndexCache(ttl_seconds=60.0)
    await prewarm_all(settings, cache)

    # Both tenants warmed for both entity types -> 4 cache entries
    assert len(cache._values) == 4  # noqa: SLF001
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_prewarm.py -v`
Expected: ImportError on `phonetics_engine.prewarm`.

- [ ] **Step 3: Write `src/phonetics_engine/prewarm.py`**

```python
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
```

- [ ] **Step 4: Wire prewarm into the FastAPI lifespan in `src/phonetics_engine/main.py`**

```python
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from phonetics_engine.config import Settings
from phonetics_engine.index_cache import IndexCache
from phonetics_engine.prewarm import prewarm_all
from phonetics_engine.routes import health, match, reload as reload_route


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    cache: IndexCache = app.state.index_cache
    if settings.phx_prewarm_enabled:
        app.state.prewarm_task = asyncio.create_task(prewarm_all(settings, cache))
    yield
    task = getattr(app.state, "prewarm_task", None)
    if task and not task.done():
        task.cancel()


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="phonetics-engine", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.index_cache = IndexCache(ttl_seconds=float(settings.phx_cache_ttl_seconds))
    app.include_router(health.router)
    app.include_router(match.router)
    app.include_router(reload_route.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run — PASS**

Run: `uv run pytest tests/integration/test_prewarm.py -v`
Expected: 1 passed.

- [ ] **Step 6: Verify other tests still pass**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/phonetics_engine/prewarm.py src/phonetics_engine/main.py tests/integration/test_prewarm.py
git commit -m "feat(prewarm): startup background task warming all known tenants"
```

---

## Phase 13 — PII-aware logging + metrics

**Goal:** INFO-level logs contain no PII; X-Internal-Token never appears in logs; Prometheus metrics exposed at `GET /metrics`.

**Files:**
- Create: `src/phonetics_engine/logging_setup.py`, `src/phonetics_engine/metrics.py`, `src/phonetics_engine/routes/metrics.py`
- Modify: `src/phonetics_engine/routes/match.py` (instrument), `src/phonetics_engine/main.py` (call setup)
- Test: `tests/integration/test_logging_pii.py`

### Task 13.1: Logging setup with PII redact

- [ ] **Step 1: Failing test `tests/integration/test_logging_pii.py`**

```python
import logging

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    monkeypatch.setenv("PHX_LOG_PAYLOAD", "0")
    return TestClient(create_app())


@respx.mock
def test_info_log_contains_no_pii(client, caplog):
    respx.get("https://test.supabase.co/rest/v1/companies").mock(
        return_value=httpx.Response(200, json=[
            {"id": "c1", "display_name": "Waysis", "canonical_name": "waysis", "aliases": []},
        ])
    )
    with caplog.at_level(logging.INFO, logger="phonetics_engine"):
        client.post(
            "/v1/match",
            headers={"X-Internal-Token": "secret"},
            json={"query": "Wasteless Inc", "entity_type": "company", "customer_id": "tenant-x"},
        )

    text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Wasteless Inc" not in text  # query: PII
    assert "Waysis" not in text  # display_name: PII
    assert "secret" not in text  # never log the token
    assert "tenant-x" in text  # customer_id is fine
    assert "single_high_confidence" in text or "no_match" in text or "exact" in text
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_logging_pii.py -v`
Expected: assertions fail because no logging is configured yet.

- [ ] **Step 3: Write `src/phonetics_engine/logging_setup.py`**

```python
import logging
import sys


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
```

- [ ] **Step 4: Instrument the match handler in `src/phonetics_engine/routes/match.py`**

After computing `decision`, log a non-PII summary; never log `query` or `display_name` at INFO. Add at the top:

```python
import time
```

Replace the body of `match()` with the instrumented version (only the new log lines are shown — keep the rest):

```python
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

    return MatchResponse(...)  # unchanged
```

- [ ] **Step 5: Call `configure_logging()` from `create_app` in `src/phonetics_engine/main.py`**

```python
from phonetics_engine.logging_setup import configure_logging

def create_app() -> FastAPI:
    configure_logging()
    settings = Settings()
    ...
```

- [ ] **Step 6: Add request-token redaction filter — Pydantic and FastAPI never log headers by default, so no extra work for token redaction unless a custom middleware logs them; the assertion `"secret" not in text` passes by default. If you add request-logging middleware later, ensure it strips `X-Internal-Token` from header dumps.**

(No code change needed; assertion confirms.)

- [ ] **Step 7: Run — PASS**

Run: `uv run pytest tests/integration/test_logging_pii.py -v`
Expected: 1 passed.

- [ ] **Step 8: Commit**

```bash
git add src/phonetics_engine/logging_setup.py src/phonetics_engine/routes/match.py src/phonetics_engine/main.py tests/integration/test_logging_pii.py
git commit -m "feat(logging): PII-aware INFO logs at match-handler"
```

### Task 13.2: Prometheus metrics

- [ ] **Step 1: Write `src/phonetics_engine/metrics.py`**

```python
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "phx_match_requests_total",
    "Total /v1/match requests",
    ["customer_id", "entity_type", "decision"],
)

LATENCY = Histogram(
    "phx_match_latency_seconds",
    "Match latency in seconds",
    ["customer_id", "entity_type", "stage"],  # stage: total | espeak | faiss
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
)

INDEX_BUILD = Histogram(
    "phx_index_build_seconds",
    "Time spent building a tenant index",
    ["customer_id", "entity_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
)

SERVICE_ERRORS = Counter(
    "phx_service_errors_total",
    "Service-level errors caught and converted to decision=service_error",
    ["reason"],
)
```

- [ ] **Step 2: Add `GET /metrics` route — `src/phonetics_engine/routes/metrics.py`**

```python
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

- [ ] **Step 3: Mount + instrument**

In `src/phonetics_engine/main.py`:

```python
from phonetics_engine.routes import health, match, metrics as metrics_route, reload as reload_route
# ...
app.include_router(metrics_route.router)
```

Add to imports at the top of `src/phonetics_engine/routes/match.py`:

```python
from phonetics_engine.metrics import INDEX_BUILD, LATENCY, REQUESTS, SERVICE_ERRORS
```

Wrap the `builder` closure to instrument index-build time:

```python
            async def builder() -> TenantIndex:
                t0 = time.perf_counter()
                if req.entity_type == EntityType.COMPANY:
                    idx = await _build_company_tenant_index(req, s)
                else:
                    idx = await _build_employee_tenant_index(req, s)
                INDEX_BUILD.labels(req.customer_id, req.entity_type.value).observe(
                    time.perf_counter() - t0
                )
                return idx
```

After the `logger.info(...)` line (before `return MatchResponse(...)`), add:

```python
REQUESTS.labels(req.customer_id, req.entity_type.value, decision.value).inc()
LATENCY.labels(req.customer_id, req.entity_type.value, "total").observe(elapsed_ms / 1000.0)
```

And in the `except` block (just before the `return MatchResponse(...)` for service_error):

```python
SERVICE_ERRORS.labels("unhandled").inc()
```

- [ ] **Step 4: Smoketest `/metrics`**

Add to `tests/integration/test_match_endpoint.py`:

```python
def test_metrics_endpoint_exposes_counters(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "phx_match_requests_total" in r.text
```

- [ ] **Step 5: Run — PASS**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/phonetics_engine/metrics.py src/phonetics_engine/routes/metrics.py src/phonetics_engine/main.py src/phonetics_engine/routes/match.py tests/integration/test_match_endpoint.py
git commit -m "feat(metrics): Prometheus counters/histograms + GET /metrics"
```

---

## Phase 14 — Backwards-compat /search shim

**Goal:** Endpoint that mirrors Bonove's `/search` exactly: takes `{"name": "...", "top_k": N, "min_score": F}`, returns the old-shape response from `medewerkers_bellijst`. Independent of the new pipeline.

**Files:**
- Create: `src/phonetics_engine/routes/legacy.py`
- Modify: `src/phonetics_engine/main.py`
- Test: `tests/integration/test_legacy_search.py`

### Task 14.1: /search shim

The shim queries the existing `medewerkers_bellijst` table directly (the seed data still lives there) and returns the Bonove-compatible response.

- [ ] **Step 1: Failing test `tests/integration/test_legacy_search.py`**

```python
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from phonetics_engine.main import create_app


pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "k")
    monkeypatch.setenv("PHX_INTERNAL_TOKEN", "secret")
    monkeypatch.setenv("PHX_PREWARM_ENABLED", "0")
    return TestClient(create_app())


@respx.mock
def test_legacy_search_returns_bonove_shape(client):
    respx.get("https://test.supabase.co/rest/v1/medewerkers_bellijst").mock(
        return_value=httpx.Response(200, json=[
            {"id": "1", "voornaam": "Max",     "telefoonnummer": "31621449795", "company_name": "waysis"},
            {"id": "2", "voornaam": "Steven",  "telefoonnummer": "31621200435", "company_name": "tmc"},
        ])
    )

    r = client.post(
        "/search",
        headers={"Authorization": "Bearer legacy-token"},
        json={"name": "Max", "top_k": 3, "min_score": 0.3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "matches" in body
    assert "query_phonemes" in body
    assert body["source"] == "supabase"
    assert body["matches"][0]["name"] == "Max"
    assert body["matches"][0]["company"] == "waysis"
    assert body["matches"][0]["phone"] == "31621449795"
```

- [ ] **Step 2: Run — fail**

Run: `uv run pytest tests/integration/test_legacy_search.py -v`
Expected: 404 on `/search`.

- [ ] **Step 3: Write `src/phonetics_engine/routes/legacy.py`**

```python
from functools import lru_cache

import httpx
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

from phonetics_engine.config import Settings
from phonetics_engine.loader import _headers
from phonetics_engine.phonetics import PhoneticIndex, phonemize_name


router = APIRouter()


class LegacyRequest(BaseModel):
    name: str
    top_k: int = 3
    min_score: float = 0.3


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


async def _fetch_legacy_rows() -> list[dict]:
    s = _settings()
    url = f"{s.supabase_url}/rest/v1/medewerkers_bellijst"
    params = {"select": "voornaam,company_name,telefoonnummer,id"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params=params, headers=_headers(s))
        r.raise_for_status()
        return r.json()


@router.post("/search")
async def legacy_search(req: LegacyRequest, authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    rows = await _fetch_legacy_rows()
    names = [row["voornaam"] for row in rows]
    metadata = [{"company": row["company_name"], "phone": row["telefoonnummer"]} for row in rows]

    index = PhoneticIndex(names)
    raw = index.search(req.name, top_k=req.top_k)
    out = []
    for r in raw:
        if r["score"] < req.min_score:
            continue
        meta = metadata[names.index(r["name"])]
        out.append({
            "name": r["name"],
            "score": r["score"],
            "phonemes": r["phonemes"],
            "company": meta["company"],
            "phone": meta["phone"],
        })
    return {
        "matches": out,
        "query_phonemes": phonemize_name(req.name),
        "source": "supabase",
    }
```

- [ ] **Step 4: Mount in `src/phonetics_engine/main.py`**

```python
from phonetics_engine.routes import health, match, legacy, metrics as metrics_route, reload as reload_route
# ...
app.include_router(legacy.router)
```

- [ ] **Step 5: Run — PASS**

Run: `uv run pytest tests/integration/test_legacy_search.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/phonetics_engine/routes/legacy.py src/phonetics_engine/main.py tests/integration/test_legacy_search.py
git commit -m "feat(legacy): /search shim against medewerkers_bellijst"
```

---

## Phase 15 — Deploy to Render + smoketest

**Goal:** Service deployed to Render staging, env vars populated, `/health` 200, `/v1/match` returns a real decision against `xpots-dev` tenant data.

**Files:** none new (only verification + commit)

### Task 15.1: Push to GitHub + connect Render

- [ ] **Step 1: Verify all tests green**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 2: Verify ruff is clean**

Run: `uv run ruff check src tests`
Expected: no issues.

- [ ] **Step 3: Push to GitHub**

```bash
gh repo create phonetics-engine --private --source=. --remote=origin --push
```

- [ ] **Step 4: Connect Render to the repo**

Use the Render MCP `mcp__render__create_web_service` with:
- repo URL from previous step
- region: frankfurt
- runtime: docker
- env vars from `render.yaml` (set the secrets `SUPABASE_URL`, `SUPABASE_KEY`, `PHX_INTERNAL_TOKEN` to real values for the staging Supabase project)

- [ ] **Step 5: Wait for deploy + smoketest /health**

```bash
curl -s https://phonetics-engine.onrender.com/health
```

Expected: `{"status":"ok"}`.

- [ ] **Step 6: Smoketest /v1/match against xpots-dev**

```bash
curl -s -X POST https://phonetics-engine.onrender.com/v1/match \
  -H 'Content-Type: application/json' \
  -H 'X-Internal-Token: <prod-token>' \
  -d '{"query":"Waysis","entity_type":"company","customer_id":"xpots-dev"}'
```

Expected: HTTP 200 with `decision` ∈ {`exact`, `single_high_confidence`} and `matches[0].canonical_name == "waysis"`.

- [ ] **Step 7: Smoketest /v1/match employee tussenvoegsel**

```bash
curl -s -X POST https://phonetics-engine.onrender.com/v1/match \
  -H 'Content-Type: application/json' \
  -H 'X-Internal-Token: <prod-token>' \
  -d '{"query":"de Vries","entity_type":"employee","customer_id":"xpots-dev"}'
```

Expected: matches Sanne de Vries with `matched_field == "last_name_with_infix"`.

- [ ] **Step 8: Smoketest /search backwards-compat**

```bash
curl -s -X POST https://phonetics-engine.onrender.com/search \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <legacy-token>' \
  -d '{"name":"Max","top_k":3,"min_score":0.3}'
```

Expected: Bonove-shape response with phone in matches.

- [ ] **Step 9: Smoketest /metrics**

```bash
curl -s https://phonetics-engine.onrender.com/metrics | head -20
```

Expected: Prometheus exposition format with `phx_match_requests_total` already non-zero (from previous smoketests).

- [ ] **Step 10: Commit final + tag**

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## Self-review checklist (run before declaring v1 done)

- [ ] All `GOAL.md §9` Definition-of-Done items satisfied.
- [ ] No file in `src/phonetics_engine/` exceeds 200 lines.
- [ ] `uv run pytest -v` green.
- [ ] `uv run ruff check` green.
- [ ] OpenAPI `/openapi.json` returns the full schema; `EmployeeCandidate.phone` is rejected (422 in tests).
- [ ] p95 latency under 1500 ms on staging with 100 employees and 5 companies (manual `ab` or `wrk` run).
- [ ] Carla state-machine plan Phase 8 can call this service against staging without changes to its `phonetics_match()` adapter.

---

## Notes on TDD discipline (read before each task)

1. Tests come first. `pytest -v` should always be the second-to-last command in a step pair (test, then implement).
2. Don't widen tests after implementing. If a test was wrong, fix the test in a separate commit.
3. One commit per task minimum. `git log --oneline` should read like the plan.
4. If a step describes code, paste the exact code shown — don't paraphrase.
5. If you hit a problem the plan doesn't cover, stop and ask. Don't invent an approach.
