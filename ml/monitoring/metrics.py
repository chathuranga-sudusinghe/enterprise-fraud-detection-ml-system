from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total number of API requests"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of API requests"
)