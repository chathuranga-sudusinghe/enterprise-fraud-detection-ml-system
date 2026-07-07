# API Latency Baseline Benchmark

This directory contains non-production benchmark tooling for the current fraud
prediction API. The harness measures the system as it works now; it does not
change API behavior, feature engineering, artifacts, thresholds, Kafka, or
logging.

## What It Measures

- External end-to-end `POST /predict` latency.
- p50, p95, p99, mean, min, and max latency.
- Throughput in successful requests per second.
- Request count, successes, failures, and error rate.
- Harness process CPU time, CPU utilization estimate, and Python memory
  allocation peaks. In the default in-process mode, this includes the app work
  running inside the benchmark process. In `--url` mode, it measures the client
  harness process, not the remote server process.
- Internal handler-equivalent phases:
  - Pydantic request validation
  - one-row pandas DataFrame construction
  - feature transformation
  - persisted 445-feature alignment
  - LightGBM inference
  - threshold and response construction
  - synchronous JSONL metrics write
  - Prometheus counter and latency observation

The benchmark intentionally keeps synchronous JSONL metrics writing enabled.
Those writes go through the existing `artifacts/metrics/api_metrics.jsonl` path.

## Output Location

`artifacts/benchmarks/` is not ignored by Git in the current repository, so the
benchmark defaults to writing JSON results under the system temporary directory:

```text
<temp>/enterprise_fraud_detection_benchmarks/
```

Use `--no-output` to print results without writing a result file, or pass
`--output` to choose an explicit path outside tracked locations.

## Safe Validation Run

```bash
python tests/benchmarks/api_latency_baseline.py --preset safe --no-output
```

## Full Baseline Run

```bash
python tests/benchmarks/api_latency_baseline.py --preset full
```

The full preset runs concurrency levels `1, 5, 10, 25, 50`, with warm-up
requests and repeated runs.

## Running Against A Live API

By default, the benchmark uses FastAPI `TestClient` in-process. To measure a
running API over HTTP, start the API separately and pass:

```bash
python tests/benchmarks/api_latency_baseline.py --url http://127.0.0.1:8000 --preset full
```

Kafka is not called by the current `/predict` path and is not started by this
benchmark.

