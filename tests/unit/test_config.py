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
