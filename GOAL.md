---
title: GOAL — phonetics-engine v1 (rebuild voor Carla)
date: 2026-05-06
status: ready-for-plan
audience: implementatie-team (jij + subagents); doel is helder genoeg om een writing-plan op te starten
---

# GOAL — wat de nieuwe phonetics-engine moet worden

## 1. Eén-zinsdoel

Een **multi-tenant, intern-only matching-service** die voor Carla (Pipecat-bot bij parkeerintercom) razendsnel beslist of een uitgesproken bedrijfs- of medewerkernaam matcht met de records van één specifieke klant in een Supabase-DB die de single source of truth vormt, en die beslissing zelf classificeert (`exact` / `single_high_confidence` / `ambiguous` / `no_match` / `service_error`) zodat de bot deterministisch verder kan zonder eigen drempel-logica.

## 2. Waarom een nieuwe engine, niet aanpassen?

De huidige `Bonove/phonetics-service` doet één ding goed (Nederlandstalige fonetische similarity met espeak-ng + FAISS), maar is qua contract en datamodel fundamenteel **niet geschikt** voor Carla v1.0:

| Probleem in de huidige service | Waarom dit Carla blokkeert |
|---|---|
| **Geen `customer_id`-scoping** — alle medewerkers van alle klanten zitten in één globale FAISS-index (`main.py:25-62`, geen filter in select). | Carla bedient meerdere parkeerterreinen. Cross-tenant phone-leakage is een hard veiligheidsprobleem. |
| **`phone` zit in de response** (`main.py:126`). | Spec-eis: phone hoort bij Carla/n8n, niet bij phonetics. Regressietest `test_phonetics_request_strips_phone_field` is verplicht (state-machine spec, regel 1535). |
| **Discriminator company-vs-employee zit in `phone==""`** (`main.py:43-59`) — magic-string-trick. | Carla heeft expliciete `entity_type: "company" \| "employee"` nodig, met aparte candidate-shapes en match-fields. |
| **Geen decision-logic server-side** — service geeft ruwe scores, drempels zitten bij elke caller los. | Spec-eis: server beslist `exact / single_high_confidence / ambiguous / no_match / service_error`. Drift voorkomen tussen Carla, n8n en toekomstige clients. |
| **Geen `match_fields`-ondersteuning** — DB heeft alleen `voornaam`, geen `achternaam` / `volledige_naam` als losse FAISS-vector. Carla matcht primair op **achternaam**. | Carla matcht employees op last_name en full_name en moet `matched_field`/`matched_value` terugkrijgen voor TTS-bevestiging ("u zoekt mevrouw De Vries?"). |
| **Auth = `Authorization: Bearer` (`main.py:18, 111-113`).** | Spec wil `X-Internal-Token` voor interne service-naar-service calls. |
| **`/health` lekt alle namen** (`main.py:135-142`, `"names": _raw_names`). | Acceptabel in v0; onacceptabel in v1 multi-tenant. |
| **Geen `margin_to_next`** in response. | Carla berekent ambiguity op `margin_to_next`. Server-side beter dan client-side. |
| **Sync = pull-at-startup + handmatige `/reload`.** Geen cron, geen webhook. | Carla mag geen stale-window tegenkomen voor een net aangenomen medewerker. |

Een rebuild is dus geen "nice to have refactor", maar volgt logisch uit de security-, multi-tenant- en contract-eisen die Carla v1.0 stelt. De fonetische kern (`PhoneticIndex`, `_phonemes_to_vector` uit Bonove) blijft hergebruikbaar als bibliotheek-laag.

## 3. Wat de nieuwe engine **wel** is — scope

### 3.1 Het hoofd-endpoint

```
POST /v1/match
Headers: X-Internal-Token: <secret>
```

**Request:**
```json
{
  "query": "wasteless",
  "entity_type": "company" | "employee",
  "customer_id": "1000435",
  "scope": {"company_id": "<uuid>"} | null,
  "match_fields": ["last_name","full_name"] | null,
  "thresholds": {
    "min_match": 0.55,
    "high_confidence": 0.86,
    "ambiguity_margin": 0.12
  } | null,
  "candidates": [ /* optionele override; default uit DB */ ] | null,
  "top_k": 5
}
```

- **`candidates` is optioneel** (override-mode voor tests / edge cases). Default: service queryt zelf de DB op `customer_id` (en `scope.company_id` voor employees).
- **`thresholds` is optioneel** (per-call override). Default: env-vars per `entity_type`. Beschikbaar vanaf v1 dag 1 zodat we live kunnen kalibreren zonder redeploy.
- **`match_fields` defaults**:
  - `entity_type=company` → `["display_name", "canonical_name"]`
  - `entity_type=employee` → `["last_name", "full_name"]` (last_name primair; tussenvoegsel-variant wordt automatisch meegenomen — zie §3.3)

**Response (company):**
```json
{
  "entity_type": "company",
  "decision": "single_high_confidence",
  "applied_thresholds": {"min_match": 0.55, "high_confidence": 0.82, "ambiguity_margin": 0.10},
  "matches": [
    {"id": "<uuid>", "display_name": "Waysis", "canonical_name": "waysis",
     "score": 0.91, "margin_to_next": 0.40}
  ]
}
```

**Response (employee) — extra velden:**
```json
{
  "entity_type": "employee",
  "decision": "ambiguous",
  "applied_thresholds": {"min_match": 0.55, "high_confidence": 0.86, "ambiguity_margin": 0.12},
  "matches": [
    {"id": "<uuid>", "display_name": "Sanne de Vries", "canonical_name": "sanne de vries",
     "score": 0.78, "margin_to_next": 0.04,
     "matched_field": "last_name", "matched_value": "de Vries"}
  ]
}
```

`applied_thresholds` zit altijd in de response zodat caller en logs achteraf weten welke drempels gebruikt zijn (debug-vriendelijk; voorkomt dat een caller stilletjes kalibreert met afwijkende drempels).

**Geen `phone` in request of response.** Phone hoort uitsluitend bij Carla/n8n; phonetics weet er niet van. OpenAPI markeert `phone` als forbidden veld op `EmployeeCandidate` in request → 422 bij aanwezigheid (geen silent acceptance).

### 3.2 Decision-classificatie (server-side, autoritatief)

| Decision | Wanneer |
|---|---|
| `exact` | `query` (NFKD-genormaliseerd, lowercase, strip) gelijk aan `canonical_name` of `display_name` van **precies één** kandidaat. ≥2 exact matches → `ambiguous`, niet `exact`. |
| `single_high_confidence` | `best.score ≥ HIGH_CONFIDENCE` **én** `margin_to_next ≥ AMBIGUITY_MARGIN`. |
| `ambiguous` | `top.score - runner_up.score < AMBIGUITY_MARGIN` (meerdere kandidaten dichtbij). |
| `no_match` | `best.score < MIN_MATCH`. |
| `service_error` | Onverwachte fout, parse-error, downstream timeout, DB-fail. **Altijd** vangen — Carla mag niet crashen op 5xx. |

**Default drempels** (env-vars, per-entity_type, overschrijfbaar via `request.thresholds`):

| Drempel | Company | Employee | Rationale |
|---|---|---|---|
| `MIN_MATCH` | 0.55 | 0.55 | Onder dit niveau is signaal vrijwel ruis. |
| `HIGH_CONFIDENCE` | 0.82 | 0.86 | Employee strenger: last-name overlap is groter. |
| `AMBIGUITY_MARGIN` | 0.10 | 0.12 | Employee strenger om dezelfde reden. |

Env-vars: `PHX_COMPANY_MIN_MATCH`, `PHX_COMPANY_HIGH_CONFIDENCE`, `PHX_COMPANY_AMBIGUITY_MARGIN`, idem `PHX_EMPLOYEE_*`. Drempels worden gekalibreerd op echte intercom-audio na go-live.

**HTTP status-code policy.** `/v1/match` retourneert **altijd HTTP 200**, behalve:

| Status | Wanneer |
|---|---|
| `401` | `X-Internal-Token` ontbreekt of fout. |
| `422` | Request schema invalid (bv. `phone` in `EmployeeCandidate`, onbekend `entity_type`, ontbrekend `customer_id`). |
| `200` | **Alle** andere uitkomsten — inclusief Supabase down, espeak crash, parse-error in DB-row, downstream timeout, lege tenant-set. |

Onverwachte fouten → `200 OK` met `decision: "service_error"` + `matches: []`. Lege DB voor tenant → `200 OK` met `decision: "no_match"` + `matches: []` (geen `404`).

Reden: Carla heeft per state expliciete logica voor `service_error` (eigen TTS, `terminal_deny:service_error` afhandeling). Een `5xx` zou Carla in een retry-loop duwen die de bot vastzet voor de beller. Status-code wordt onderdeel van de OpenAPI-contract-test.

### 3.3 Multi-tenant model — DB is single source of truth

- **Elke row** in de datalaag heeft een verplichte `customer_id`. Hard-isolatie: een request met `customer_id=X` mag nooit candidates uit `customer_id=Y` zien — verplichte regressietest.
- **DB-schema** (zie `migrations/001_initial_schema.sql`):
  - `customers` (tenants, id is text-string zoals Carla het op de wire stuurt)
  - `companies` (per tenant; `display_name`, `canonical_name`, `aliases[]`)
  - `employees` (per tenant; `first_name`, `infix`, `last_name`, `full_name` als generated column)
  - `employee_company_roles` (junction, many-to-many; `phone` zit hier — phone is per rol, niet per persoon)
- **Index-laag (in-memory FAISS)**: één index per `(customer_id, entity_type [, company_id])`, lazy-loaded uit DB, gecached met **TTL 60s**. Bij cache-miss queryt de service de DB, phonemizet de candidates, bouwt FAISS-index, cachet.
- **Tussenvoegsel-matching voor employees**: `infix` is in DB één string (geen split op spaties — anders verlies je "van der" / "van den"); leading/trailing whitespace strippen. Bij build van de employee-index worden **twee FAISS-vectoren per employee** gebouwd — één op `last_name`, één op `infix + " " + last_name` (alleen als `infix` non-empty). Bij scoring kiest de service de hoogste van de twee, en `matched_field` reflecteert welke (`last_name` of `last_name_with_infix`).
- **Volledige reload**: `keep it simple` — bij cache-miss / TTL-expiry haalt de service de complete tenant-set opnieuw op. Geen incremental sync. Realistisch want we verwachten ≤10.000 medewerkers per tenant.
- **`POST /v1/reload`**: handmatige cache-flush per tenant. Body: `{"customer_id": "1000435", "entity_type": "company" | "employee" | null}` — `entity_type` weglaten = customer-wide flush (handige nood-knop). Geauthenticeerd via `X-Internal-Token`. Voor v1 niet automatisch getriggerd door Supabase — we vertrouwen op TTL.
- **`candidates` override**: alleen voor unit-tests en debug. Productie-flow van Carla stuurt geen `candidates` mee.

### 3.4 SLO's (overgenomen uit Carla state-machine spec)

| Metriek | Doel |
|---|---|
| `p95` end-to-end `/v1/match` latency | **< 1500 ms** (Carla `SETUP_LATENCY_BUDGET_MS`) |
| `p99` end-to-end | **< 3000 ms** |
| Carla-kant timeout | 5 s hard, 3 s soft (loading_employees) — service mag geen 504 worden |
| Cache | TTL 60s op FAISS-index per `(customer_id, entity_type [, company_id])` |

### 3.4.1 Cold-start mitigatie & concurrency

Cold-call performance: bij 10.000 employees kost batched espeak-phonemize ca. 1-3s. Dat is binnen `p99` maar buiten `p95`. Mitigatie:

- **Pre-warm bij startup.** Service-startup leest tenant-lijst uit Supabase, bouwt in een achtergrond-task (niet blokkerend voor `/health`) FAISS-indexen voor alle bekende `(customer_id, entity_type)`-paren. Eerste echte request krijgt warme cache.
- **Lazy fallback.** Als pre-warm faalt of een nieuwe tenant verschijnt na startup, valideert/bouwt de eerste call de cache on-demand. Pre-warm is optimalisatie, geen vereiste voor correctheid.

**Concurrency — thundering-herd op cold-cache voorkomen:**

- Per cache-key `(customer_id, entity_type [, company_id])` één `asyncio.Lock`.
- Bij parallelle cache-misses bouwt één coroutine de FAISS-index; andere coroutines wachten op completion via de lock i.p.v. zelf opnieuw te phonemize'n.
- Lock wordt vrijgegeven zodra de index in cache staat — wachters kopiëren de cache-hit zonder herbouw.

Dit is cruciaal: zonder lock zou een burst van 5 parallelle Carla-calls op een net-geëxpireerde cache 5× espeak-phonemize draaien (~5-15s totaal) i.p.v. 1× (~1-3s).

### 3.5 Auth & security

- `X-Internal-Token` header (single secret, env-config, rotatie buiten scope v1).
- TLS terminatie op edge (Render).
- **Geen** `phone`, **geen** persoonlijke records in `/health`, **geen** debug-endpoints in productie.
- Tenant-isolatie regressietest in CI verplicht.
- OpenAPI strict op `EmployeeCandidate.phone` → 422 bij aanwezigheid (regressietest).

### 3.5.1 Logging-discipline & observability (PII-aware)

GDPR + multi-tenant: rauwe namen of queries mogen niet ongeconditioneerd in logs verschijnen.

**Log-levels:**

- **`INFO`**: alleen `customer_id`, `entity_type`, `decision`, `score` (top-1), `margin_to_next`, `latency_ms`, cache `hit`/`miss`, gebruikte threshold-source (`env` of `request`). **Geen** `query`-tekst, **geen** `display_name`, **geen** `matched_value` — die zijn PII.
- **`DEBUG`**: mag full payload (query + matched values), alleen aan via env-var `PHX_LOG_PAYLOAD=1` voor staging-debugging.
- **`X-Internal-Token`** wordt gefilterd uit alle log-output op middleware-niveau (request-logger redact-list).

**Metrics** (Prometheus / OpenTelemetry):

- `phx_match_requests_total{customer_id, entity_type, decision}` — counter.
- `phx_match_latency_seconds{customer_id, entity_type, stage}` — histogram met `stage` ∈ {`espeak`, `faiss`, `total`}, zodat we phonemize-cost los kunnen zien van vector-search-cost.
- `phx_cache_hit_ratio{customer_id, entity_type}` — gauge (rollende window).
- `phx_index_build_seconds{customer_id, entity_type}` — histogram (cold-start kosten per tenant).
- `phx_service_errors_total{reason}` — counter, alarm bij stijging.

Decision-distribution (`phx_match_requests_total` met `decision`-label) is direct bruikbaar als drempel-kalibratie-data: als `ambiguous`-rate > 20% op een tenant, drempels te streng; als `service_error`-rate > 1%, infra-probleem.

### 3.6 Backwards-compatibility

- **Oude `/search` endpoint** uit Bonove blijft ongewijzigd werken voor bestaande callers (n8n proxies). Geen wijziging in response-shape.
- **Oude `medewerkers_bellijst` tabel** blijft staan in Supabase — niet droppen totdat `/search`-shim sunset is. De nieuwe service raakt deze tabel niet aan.
- Carla switcht direct naar `/v1/match` op de nieuwe schema-tabellen.
- Deprecation-banner in OpenAPI; sunset-date wordt los gepland.

## 4. Wat de nieuwe engine **niet** is — anti-scope

- **Geen LLM-fallback.** De engine is deterministisch. Als phonemizer/FAISS faalt → `service_error`, geen creatieve guesses.
- **Geen authn-laag voor eindgebruikers.** Internal-only.
- **Geen UI.** Gewoon een service.
- **Geen sync-orchestratie buiten de eigen DB.** Xpots levert data in Supabase (via Bonove-tooling); de koppeling Xpots → Supabase staat **buiten dit project**. Wij leveren `/v1/reload` als handmatige cache-flush.
- **Geen real-time updates.** TTL 60s + handmatige reload zijn de enige refresh-mechanismen in v1. Database-webhook → reload kan v1.x worden.
- **Geen incremental reload.** Bij cache-miss laden we de hele tenant-set opnieuw — voldoende voor ≤10k records.
- **Geen taal-detectie.** Default `nl` (espeak-ng). Andere talen later.
- **Geen pricing/quota/rate-limit.** Internal-only verkeer; later.
- **Geen aliases voor employees** in v1 (companies hebben wel aliases). Eventueel v1.x.

## 5. Architectuurschets

```
Carla (Pipecat)
   │  POST /v1/match  (X-Internal-Token)
   │  body: { query, entity_type, customer_id, scope?, thresholds?, ... }
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  phonetics-engine (FastAPI)                                         │
│  ├─ routes/match.py        ── Pydantic v2 models, OpenAPI strict    │
│  ├─ routes/reload.py       ── POST /v1/reload (cache flush)         │
│  ├─ routes/legacy.py       ── /search shim (backwards-compat)       │
│  ├─ services/decision.py   ── classify(matches, applied_thresholds) │
│  ├─ services/matcher.py    ── FAISS search + tussenvoegsel-logic    │
│  ├─ services/index_cache.py── per-(tenant, entity) FAISS, TTL 60s   │
│  ├─ services/loader.py     ── Supabase pull (companies/employees)   │
│  ├─ services/phonetics.py  ── (hergebruik PhoneticIndex uit Bonove) │
│  └─ middleware/auth.py     ── X-Internal-Token verify               │
└─────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
            Supabase Postgres (SSOT)
            ├─ customers
            ├─ companies               (customer_id, display_name, canonical_name, aliases[])
            ├─ employees               (customer_id, first_name, infix, last_name, full_name*)
            └─ employee_company_roles  (employee_id, company_id, phone)   ← phone hier
```

`*full_name` = generated stored column.

## 6. Werkschatting

**~6 mensdagen** (range 5-7).

| Onderdeel | Dagen |
|---|---|
| DB-migratie (al beschikbaar als `migrations/001_initial_schema.sql`) + verificatie | 0.25 |
| Loader (Supabase pull → in-memory shape) + tussenvoegsel-vectoren | 1.0 |
| Index-cache met TTL + `/v1/reload` | 0.75 |
| `/v1/match` route + Pydantic + auth + threshold-override | 1.0 |
| Decision-logic + `margin_to_next` + exact-tie handling | 0.5 |
| Tests (unit + tenant-isolation + phone-strip + threshold-override regressie) | 1.25 |
| Backwards-compat shim (`/search`) + smoketest | 0.25 |
| Deploy (Render) + verificatie + cutover | 0.5 |
| Buffer (espeak edge cases, code review, OpenAPI publish) | 0.5 |
| **Totaal** | **~6** |

## 7. Aanbeveling

**Optie B — nieuwe `/v1/match` bouwen, `/search` als shim laten staan, DB-schema vernieuwen.** Reden:
1. Optie A (Carla past zich aan oud schema aan) lost de cross-tenant leak en de phone-leak niet op — die zitten in het oude endpoint en datamodel zelf.
2. Server-side decisions voorkomt logica-drift over meerdere clients.
3. DB als SSOT met multi-tenant-tabellen geeft een stabiele basis voor groei (10k records per tenant).
4. Migratiekosten zijn beheersbaar (~6 dagen) en `/search` blijft draaien voor andere clients.

## 8. Open punten voor het writing-plan

- **Espeak-process pooling**: phonemizer forkt een proces per call — pre-warmed pool of one-shot? Bepaalt of cold-call binnen p95-budget past zonder de pre-warm-strategie te overbelasten.
- **OpenAPI-publicatie**: hosted via `/openapi.json` op de service zelf, of in een aparte schema-repo voor Carla's contract-test in CI?
- **Pre-warm scope bij startup**: alle tenants of alleen "actieve" tenants (bv. laatste 24h activity)? Bij groei naar tientallen tenants wordt all-warm duur in geheugen.
- **Render deployment-shape**: single web service of split (web + reload-worker)? Voor v1 vermoedelijk single, maar bevestigen.

## 9. Definition of Done — wanneer is v1 "klaar"?

- DB-migratie `001_initial_schema.sql` toegepast op staging + productie Supabase.
- `POST /v1/match` werkt voor `entity_type=company` en `entity_type=employee` met alle 5 decisions, getest met unit + integration tests.
- Tenant-isolatie regressietest groen (request `customer_id=A` ziet nooit data van `customer_id=B`).
- Phone-strip regressietest groen (geen `phone` in request of response; 422 bij aanwezigheid).
- Threshold-override regressietest groen (request met `thresholds` overschrijft env-defaults; `applied_thresholds` correct in response).
- Tussenvoegsel-matching getest: query "Vries" en "de Vries" matchen beide op employee "Sanne de Vries", met juiste `matched_field`/`matched_value`.
- HTTP-status-policy regressietest groen: gesimuleerde Supabase-fail / espeak-fail → `200 OK` met `decision: "service_error"`, niet `5xx`.
- Concurrency-lock test groen: 5 parallelle cache-miss-requests op dezelfde key triggeren **1× index-build** (niet 5×).
- PII-logging regressietest groen: bij `PHX_LOG_PAYLOAD=0` bevatten INFO-logs geen `query`, geen `display_name`, geen `matched_value`. `X-Internal-Token` nooit in logs.
- p95 latency < 1500 ms gemeten op staging onder realistische load (10k employees per tenant).
- Carla in staging kan een end-to-end gesprek voeren: bedrijfsnaam → `single_high_confidence` → confirm → employee → confirm → call_employee. Zonder client-side drempel-logica.
- `/search` shim retourneert nog steeds zelfde response voor bestaande callers.
- `/v1/reload` flushed cache per `(customer_id, entity_type)` en logt de actie.
- Deploy gedocumenteerd (`render.yaml` + `.env.example` + secret-rotatie noot).
- OpenAPI gepubliceerd; Carla-team heeft contract-test in eigen CI.

---

**Volgende stap:** dit GOAL.md valideren, daarna `superpowers:writing-plans` invoken om een concrete TDD-implementatieplan te schrijven (vergelijkbaar opbouw als het Carla state-machine plan).
