-- =====================================================================
-- phonetics-engine v1 — initial schema
-- Multi-tenant matching: customers → companies → employees ↔ roles
-- DB is single source of truth. Carla queryt customer-scoped candidates.
-- =====================================================================

-- 1) tenants ----------------------------------------------------------
create table customers (
  id          text primary key,                  -- bv. "1000435" (matches Carla wire)
  name        text not null,                     -- "Xpots Hilversum"
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- 2) bedrijven per tenant --------------------------------------------
create table companies (
  id              uuid primary key default gen_random_uuid(),
  customer_id     text not null references customers(id) on delete cascade,
  display_name    text not null,                 -- "TaxiCentrale Maassluis"
  canonical_name  text not null,                 -- "taxicentrale maassluis" (NFKD-lower-strip)
  aliases         text[] not null default '{}',  -- ["TCM", "Maassluis Taxi"]
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  unique (customer_id, canonical_name)
);

create index idx_companies_customer on companies(customer_id);

-- 3) medewerkers per tenant (geen phone hier — die hoort bij rol) ----
create table employees (
  id           uuid primary key default gen_random_uuid(),
  customer_id  text not null references customers(id) on delete cascade,
  first_name   text not null,
  infix        text,                             -- "de", "van der"; nullable
  last_name    text not null,                    -- primaire match-target
  full_name    text generated always as (
    case
      when infix is null or length(trim(infix)) = 0
        then first_name || ' ' || last_name
      else first_name || ' ' || infix || ' ' || last_name
    end
  ) stored,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index idx_employees_customer       on employees(customer_id);
create index idx_employees_last_name_norm on employees(customer_id, lower(last_name));

-- 4) many-to-many: medewerker ↔ bedrijf, met phone per rol -----------
create table employee_company_roles (
  id           uuid primary key default gen_random_uuid(),
  employee_id  uuid not null references employees(id) on delete cascade,
  company_id   uuid not null references companies(id) on delete cascade,
  phone        text not null,                    -- E.164 zonder "+", bv. "31621449795"
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  unique (employee_id, company_id),
  check (phone ~ '^[0-9]{8,15}$')
);

create index idx_ecr_company  on employee_company_roles(company_id);
create index idx_ecr_employee on employee_company_roles(employee_id);

-- 5) updated_at trigger ----------------------------------------------
create or replace function set_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

create trigger trg_customers_updated_at  before update on customers
  for each row execute function set_updated_at();
create trigger trg_companies_updated_at  before update on companies
  for each row execute function set_updated_at();
create trigger trg_employees_updated_at  before update on employees
  for each row execute function set_updated_at();
create trigger trg_ecr_updated_at        before update on employee_company_roles
  for each row execute function set_updated_at();

-- 6) seed dev-tenant met bestaande testdata --------------------------
-- (achternamen verzonnen voor lokale tests; Xpots vervangt deze later)
insert into customers (id, name) values
  ('xpots-dev', 'Xpots Dev/Test');

with company_seed as (
  insert into companies (customer_id, display_name, canonical_name) values
    ('xpots-dev', 'Waysis',     'waysis'),
    ('xpots-dev', 'Unplugged',  'unplugged'),
    ('xpots-dev', 'Xpots',      'xpots'),
    ('xpots-dev', 'Taxameter',  'taxameter'),
    ('xpots-dev', 'TMC',        'tmc')
  returning id, canonical_name
),
employee_seed as (
  insert into employees (customer_id, first_name, infix, last_name) values
    ('xpots-dev', 'Max',      null,    'Jansen'),
    ('xpots-dev', 'Tristan',  'van',   'Doorn'),
    ('xpots-dev', 'Matthijs', null,    'Bakker'),
    ('xpots-dev', 'Steven',   null,    'de Vries')
  returning id, first_name
)
insert into employee_company_roles (employee_id, company_id, phone)
select e.id, c.id, p.phone
from (values
  ('Max',      'waysis',    '31621449795'),
  ('Tristan',  'unplugged', '31611216110'),
  ('Matthijs', 'xpots',     '31655356254'),
  ('Steven',   'taxameter', '31621200435'),
  ('Steven',   'tmc',       '31621200435')
) as p(first_name, canonical_name, phone)
join employee_seed e on e.first_name = p.first_name
join company_seed  c on c.canonical_name = p.canonical_name;

-- =====================================================================
-- NB: oude `medewerkers_bellijst` tabel blijft staan voor backwards-
-- compat met `/search`-shim. Niet droppen totdat shim sunset is.
-- =====================================================================
