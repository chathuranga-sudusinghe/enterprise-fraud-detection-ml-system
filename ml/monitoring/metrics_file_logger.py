from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# detect project root safely
PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
METRICS_FILE = METRICS_DIR / "api_metrics.jsonl"

def log_api_metric(event: Dict[str, Any]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    with METRICS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")