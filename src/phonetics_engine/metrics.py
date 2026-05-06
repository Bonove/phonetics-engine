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
