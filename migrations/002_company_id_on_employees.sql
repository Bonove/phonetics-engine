-- employees.company_id is referenced by the app (fetch_employees query +
-- PostgREST resource embedding: companies!inner(customer_id)).
-- The initial schema omitted this column; add it here.
alter table
    employees
add
    column company_id uuid references companies(id) on delete
set
    null;

create index idx_employees_company on employees(company_id);