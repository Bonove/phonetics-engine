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
