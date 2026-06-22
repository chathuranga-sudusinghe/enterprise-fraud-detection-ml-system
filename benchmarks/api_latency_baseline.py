from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
from fastapi.testclient import TestClient

import api.main as api_main
from artifacts.metrics.metrics_file_logger import log_api_metric


DEFAULT_LOAD_LEVELS = [1, 5, 10, 25, 50]
DEFAULT_REPEATS = 3
DEFAULT_REQUESTS_PER_CLIENT = 20
DEFAULT_WARMUP_REQUESTS = 10

SAFE_LOAD_LEVELS = [1]
SAFE_REPEATS = 1
SAFE_REQUESTS_PER_CLIENT = 3
SAFE_WARMUP_REQUESTS = 1

REPRESENTATIVE_INPUT = {
    "TransactionDT": 86_400,
    "TransactionAmt": 100.0,
    "card1": 1234,
    "card2": 111,
    "card3": 150,
    "card4": "visa",
    "addr1": 100,
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    return {
        "min_ms": min(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "max_ms": max(latencies_ms),
    }


def timed_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def make_payload() -> dict[str, dict[str, Any]]:
    return {"data": dict(REPRESENTATIVE_INPUT)}


def make_client(url: str | None) -> tuple[Any, Callable[[], None]]:
    if url:
        client = httpx.Client(base_url=url, timeout=30.0)
        return client, client.close

    client = TestClient(api_main.app)
    return client, client.close


def post_predict(client: Any, payload: dict[str, dict[str, Any]]) -> None:
    response = client.post("/predict", json=payload)
    response.raise_for_status()

    body = response.json()
    if "fraud_probability" not in body or "fraud_prediction" not in body:
        raise RuntimeError(f"Unexpected response schema: {body}")


def run_warmup(client: Any, warmup_requests: int) -> None:
    for _ in range(warmup_requests):
        post_predict(client, make_payload())


def run_external_repeat(
    *,
    client_factory: Callable[[], tuple[Any, Callable[[], None]]],
    concurrency: int,
    requests_per_client: int,
) -> dict[str, Any]:
    request_count = concurrency * requests_per_client
    latencies_ms: list[float] = []
    failed_requests = 0
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    tracemalloc.start()

    def client_worker() -> tuple[list[float], int]:
        client, close_client = client_factory()
        worker_latencies_ms: list[float] = []
        worker_failures = 0
        try:
            for _ in range(requests_per_client):
                try:
                    _, elapsed_ms = timed_ms(
                        lambda: post_predict(client, make_payload())
                    )
                    worker_latencies_ms.append(elapsed_ms)
                except Exception:
                    worker_failures += 1
        finally:
            close_client()

        return worker_latencies_ms, worker_failures

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(client_worker) for _ in range(concurrency)]
        for future in as_completed(futures):
            worker_latencies, worker_failures = future.result()
            latencies_ms.extend(worker_latencies)
            failed_requests += worker_failures

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    successful_requests = len(latencies_ms)

    return {
        "request_count": request_count,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate": failed_requests / request_count if request_count else 0.0,
        "throughput_rps": successful_requests / wall_seconds if wall_seconds else 0.0,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_pct": (
            (cpu_seconds / wall_seconds) * 100 if wall_seconds else 0.0
        ),
        "cpu_seconds_per_request": (
            cpu_seconds / request_count if request_count else 0.0
        ),
        "tracemalloc_current_mb": current_bytes / (1024 * 1024),
        "tracemalloc_peak_mb": peak_bytes / (1024 * 1024),
        "latency": summarize_latencies(latencies_ms),
    }


def run_external_benchmark(
    *,
    url: str | None,
    load_levels: list[int],
    repeats: int,
    requests_per_client: int,
    warmup_requests: int,
) -> list[dict[str, Any]]:
    client, close_client = make_client(url)
    try:
        run_warmup(client, warmup_requests)
    finally:
        close_client()

    def client_factory() -> tuple[Any, Callable[[], None]]:
        return make_client(url)

    results = []
    for concurrency in load_levels:
        for repeat_index in range(1, repeats + 1):
            repeat_result = run_external_repeat(
                client_factory=client_factory,
                concurrency=concurrency,
                requests_per_client=requests_per_client,
            )
            repeat_result.update(
                {
                    "concurrency": concurrency,
                    "repeat": repeat_index,
                    "requests_per_client": requests_per_client,
                    "mode": "http" if url else "in_process_testclient",
                }
            )
            results.append(repeat_result)

    return results


def run_internal_phase_once() -> dict[str, float]:
    payload = make_payload()
    phase_ms: dict[str, float] = {}
    total_start = time.perf_counter()

    transaction, phase_ms["request_validation_ms"] = timed_ms(
        lambda: api_main.TransactionInput(**payload)
    )
    _, phase_ms["prometheus_counter_ms"] = timed_ms(api_main.REQUEST_COUNT.inc)
    df, phase_ms["dataframe_construction_ms"] = timed_ms(
        lambda: pd.DataFrame([transaction.data])
    )
    transformed, phase_ms["feature_transformation_ms"] = timed_ms(
        lambda: api_main.predictor.feature_engine.transform(df)
    )
    aligned, phase_ms["feature_alignment_ms"] = timed_ms(
        lambda: transformed[api_main.predictor.feature_columns]
    )
    y_proba, phase_ms["lightgbm_inference_ms"] = timed_ms(
        lambda: api_main.predictor.model.predict_proba(aligned)[:, 1]
    )

    def build_response() -> dict[str, Any]:
        y_pred = (y_proba >= api_main.predictor.threshold).astype(int)
        result = df.copy()
        result["fraud_probability"] = y_proba
        result["fraud_prediction"] = y_pred
        return {
            "fraud_probability": float(result["fraud_probability"].iloc[0]),
            "fraud_prediction": int(result["fraud_prediction"].iloc[0]),
        }

    response, phase_ms["threshold_and_response_ms"] = timed_ms(build_response)
    _, phase_ms["jsonl_metrics_write_ms"] = timed_ms(
        lambda: log_api_metric(
            {
                "endpoint": "/predict",
                "fraud_probability": response["fraud_probability"],
                "fraud_prediction": response["fraud_prediction"],
            }
        )
    )
    elapsed_seconds = time.perf_counter() - total_start
    _, phase_ms["prometheus_latency_observe_ms"] = timed_ms(
        lambda: api_main.REQUEST_LATENCY.observe(elapsed_seconds)
    )
    phase_ms["handler_equivalent_total_ms"] = elapsed_seconds * 1000
    return phase_ms


def run_internal_phase_benchmark(iterations: int) -> dict[str, Any]:
    phase_samples: dict[str, list[float]] = {}
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    tracemalloc.start()

    for _ in range(iterations):
        sample = run_internal_phase_once()
        for name, value in sample.items():
            phase_samples.setdefault(name, []).append(value)

    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu

    return {
        "iterations": iterations,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_pct": (
            (cpu_seconds / wall_seconds) * 100 if wall_seconds else 0.0
        ),
        "cpu_seconds_per_iteration": cpu_seconds / iterations if iterations else 0.0,
        "tracemalloc_current_mb": current_bytes / (1024 * 1024),
        "tracemalloc_peak_mb": peak_bytes / (1024 * 1024),
        "phases": {
            phase_name: summarize_latencies(samples)
            for phase_name, samples in sorted(phase_samples.items())
        },
    }


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(tempfile.gettempdir()) / "enterprise_fraud_detection_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"api_latency_baseline_{timestamp}.json"


def parse_load_levels(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-production latency benchmark for the current fraud API."
    )
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "Optional running API base URL, such as http://127.0.0.1:8000. "
            "When omitted, FastAPI TestClient is used in-process."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["safe", "full"],
        default="full",
        help="Use 'safe' for a tiny validation run or 'full' for baseline scenarios.",
    )
    parser.add_argument(
        "--load-levels",
        default=None,
        help="Comma-separated concurrency levels. Overrides the preset.",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--requests-per-client", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--internal-iterations", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON result path. Defaults to a system temp directory.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print the benchmark JSON and do not write a result file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.preset == "safe":
        load_levels = SAFE_LOAD_LEVELS
        repeats = SAFE_REPEATS
        requests_per_client = SAFE_REQUESTS_PER_CLIENT
        warmup_requests = SAFE_WARMUP_REQUESTS
        internal_iterations = 3
    else:
        load_levels = DEFAULT_LOAD_LEVELS
        repeats = DEFAULT_REPEATS
        requests_per_client = DEFAULT_REQUESTS_PER_CLIENT
        warmup_requests = DEFAULT_WARMUP_REQUESTS
        internal_iterations = 30

    if args.load_levels:
        load_levels = parse_load_levels(args.load_levels)
    if args.repeats is not None:
        repeats = args.repeats
    if args.requests_per_client is not None:
        requests_per_client = args.requests_per_client
    if args.warmup_requests is not None:
        warmup_requests = args.warmup_requests
    if args.internal_iterations is not None:
        internal_iterations = args.internal_iterations

    started_at = datetime.now(timezone.utc).isoformat()
    external_results = run_external_benchmark(
        url=args.url,
        load_levels=load_levels,
        repeats=repeats,
        requests_per_client=requests_per_client,
        warmup_requests=warmup_requests,
    )
    internal_results = run_internal_phase_benchmark(internal_iterations)

    result = {
        "benchmark": "api_latency_baseline",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_behavior_changed": False,
        "kafka_called_by_predict_path": False,
        "jsonl_metrics_writing_enabled": True,
        "representative_input": REPRESENTATIVE_INPUT,
        "config": {
            "url": args.url,
            "mode": "http" if args.url else "in_process_testclient",
            "load_levels": load_levels,
            "repeats": repeats,
            "requests_per_client": requests_per_client,
            "warmup_requests": warmup_requests,
            "internal_iterations": internal_iterations,
        },
        "external_end_to_end": external_results,
        "internal_phases": internal_results,
    }

    output_json = json.dumps(result, indent=2, sort_keys=True)
    print(output_json)

    if not args.no_output:
        output_path = args.output or default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json + "\n", encoding="utf-8")
        print(f"\nWrote benchmark result to: {output_path}")


if __name__ == "__main__":
    main()
