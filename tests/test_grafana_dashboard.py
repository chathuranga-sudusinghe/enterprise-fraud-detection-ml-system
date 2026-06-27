import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = (
    PROJECT_ROOT
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "model_v2_api_dashboard.json"
)


def test_model_v2_grafana_dashboard_is_valid_and_references_active_metrics():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    dashboard_text = json.dumps(dashboard)

    assert dashboard["title"] == "Model v2 API Monitoring"
    assert "api_requests_total" in dashboard_text
    assert "api_request_latency_seconds_bucket" in dashboard_text
    assert "/predict" in dashboard_text
    assert "/predict/v2" in dashboard_text
    assert "model_version" in dashboard_text
    assert "model_family" in dashboard_text
    assert "status" in dashboard_text
