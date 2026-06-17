# Repository Architecture and Capability Audit

Date: 2026-06-16  
Branch inspected: `feature/repository-architecture-audit`  
Scope: read-only architecture and capability audit, with this report as the only created file.

## Executive summary

This repository contains a real, non-trivial fraud detection ML system with implemented components for a local lakehouse workflow, LightGBM training, persisted model artifacts, FastAPI inference, basic Kafka-to-API scoring, Airflow DAG definitions, Docker packaging, Prometheus-compatible API metrics, GitHub Actions CI, and Terraform AWS infrastructure scaffolding.

The strongest verified runtime path is:

`api.main` -> `ml.inference.predict.FraudPredictor` -> `model_artifacts/*_v1`

The active API entry point is `api/main.py`, not `api/inference.py`. The active inference implementation is `ml/inference/predict.py`. The active persisted artifacts are:

- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/feature_transformer_v1.joblib`
- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/metadata_v1.json`
- `model_artifacts/threshold_v1.json`

The repository should not be described as production-ready. Evidence supports "production-oriented architecture" or "production-style components," but several areas are partial, simulated, stale, broken, or unvalidated. Key issues include a stale `api/inference.py` path that expects a missing `.pkl` model, an Airflow retraining DAG referencing missing registry code, weak test coverage, unvalidated Terraform deployment state, local-file metrics only, no API authentication, no schema-enforced API contract beyond `data: dict`, and unresolved baseline/threshold governance.

Agentic AI and selective inference should be treated as planned only. No runtime implementation evidence was found for agentic decisioning, selective model routing, fallback inference, or autonomous action.

## Current repository structure

Top-level structure inspected:

```text
.
+-- .github/
|   +-- ISSUE_TEMPLATE/
|   +-- workflows/ci.yml
+-- api/
|   +-- main.py
|   +-- inference.py
|   +-- streaming/
+-- artifacts/
|   +-- metrics/
|   +-- runs/
+-- configs/
+-- docker/
+-- docs/
|   +-- reports/
+-- lakehouse/
|   +-- raw/
|   +-- external/
|   +-- processed/
|   +-- curated/
|   +-- splits/
|   +-- transformations/
+-- ml/
|   +-- explainability/
|   +-- inference/
|   +-- monitoring/
|   +-- pipelines/
|   +-- registry/
|   +-- training/
|   +-- utils/
+-- model_artifacts/
+-- monitoring/
+-- notebooks/
+-- observability/
+-- orchestration/
+-- scripts/
+-- terraform/
+-- tests/
```

Important root files:

- `README.md`
- `AI_USAGE.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `LICENSE`
- `requirements.txt`
- `requirements_full.txt`
- `pytest.ini`
- `docker-compose.yml`
- `.env` exists locally and was not opened during this audit.

## Active runtime architecture

Verified active API runtime:

```text
uvicorn api.main:app
  -> api/main.py
  -> FastAPI app
  -> startup/import-time FraudPredictor()
  -> ml/inference/predict.py
  -> model_artifacts/fraud_lgbm_v1.joblib
  -> model_artifacts/feature_transformer_v1.joblib
  -> model_artifacts/feature_columns_v1.json
  -> model_artifacts/threshold_v1.json
```

Evidence:

- `api/Dockerfile` runs `uvicorn api.main:app`.
- `docker/Dockerfile` runs `uvicorn api.main:app`.
- `.github/workflows/ci.yml` builds `docker/Dockerfile`.
- `docker-compose.yml` builds the API service from `api/Dockerfile`.
- `tests/test_api.py` imports `app` from `api.main`.
- `api/main.py` imports `FraudPredictor` from `ml.inference.predict`.

Active API endpoints in `api/main.py`:

- `GET /`
- `GET /health`
- `POST /predict`
- mounted Prometheus ASGI app at `/metrics`

## ML lifecycle architecture

Training path:

```text
lakehouse/splits/*.parquet
  -> ml/pipelines/training_pipeline.py
  -> FraudFeatureEngineeringEngine.fit_transform()
  -> train_lightgbm()
  -> find_optimal_threshold(target_recall=0.95)
  -> evaluate_model()
  -> model_artifacts/*_v1
  -> artifacts/runs/training_*/manifest.json
```

Feature engineering path:

- Implemented in `ml/training/feature_engineering.py`.
- Requires at least `TransactionDT`, `TransactionAmt`, `card1`, `card2`, `card3`, `card4`, and `addr1`.
- Creates time features, UID, frequency features, UID amount aggregates, and `uid_amt_deviation`.
- Freezes learned categorical levels and final feature schema in the serialized transformer.

Evaluation path:

- Implemented in `ml/training/evaluate.py`.
- Computes ROC-AUC, accuracy, precision, recall, F1, confusion matrix counts, threshold, and alert rate.
- Threshold selection is implemented in `ml/utils/threshold.py`.

Batch inference path:

```text
lakehouse/external/test_*.parquet
  -> lakehouse/transformations/process_test_batch.py
  -> lakehouse/processed/test_merged.parquet
  -> lakehouse/transformations/transform_test_batch.py
  -> lakehouse/curated/test_batch_curated.parquet
  -> lakehouse/transformations/run_batch_prediction.py
  -> lakehouse/curated/test_batch_predictions.parquet
  -> artifacts/runs/batch_*/manifest.json
```

## Artifact and feature contract

Loaded runtime artifacts:

| Artifact | Status | Evidence |
|---|---:|---|
| `model_artifacts/fraud_lgbm_v1.joblib` | Implemented / loadable | Loaded by `FraudPredictor`; artifact inspection found `lightgbm.sklearn.LGBMClassifier` |
| `model_artifacts/feature_transformer_v1.joblib` | Implemented / loadable | Loaded by `FraudPredictor`; artifact inspection found `FraudFeatureEngineeringEngine` |
| `model_artifacts/feature_columns_v1.json` | Implemented | 445 feature columns |
| `model_artifacts/metadata_v1.json` | Implemented | `best_iteration=361`, `optimal_threshold=0.008540712517184246`, `n_features=445` |
| `model_artifacts/threshold_v1.json` | Implemented | threshold `0.008540712517184246` |

Direct artifact inspection verified:

- transformer class: `ml.training.feature_engineering.FraudFeatureEngineeringEngine`
- transformer schema length: `445`
- feature column count: `445`
- transformer schema equals `feature_columns_v1.json`: `True`
- model class: `lightgbm.sklearn.LGBMClassifier`
- model feature count: `445`
- model best iteration: `361`

Important contract risk:

- The saved transformer and `feature_columns_v1.json` include `uid_time_to_next` and `uid_time_from_prev`.
- The current source in `ml/training/feature_engineering.py` does not visibly create those two fields.
- Notebook evidence in `notebooks/03_behavioral_aggregation_engine.ipynb` includes those features.
- This means the active serialized transformer preserves a wider/historical feature contract than the current source appears to reproduce.

## API and inference flow

Active FastAPI flow:

```text
POST /predict
  -> Pydantic TransactionInput(data: dict)
  -> pandas.DataFrame([transaction.data])
  -> FraudPredictor.predict(df)
  -> feature_transformer_v1.joblib.transform(df)
  -> align to feature_columns_v1.json
  -> fraud_lgbm_v1.joblib.predict_proba(...)
  -> threshold_v1.json
  -> response with fraud_probability and fraud_prediction
  -> artifacts/metrics/api_metrics.jsonl append
```

Active inference implementation:

- `ml/inference/predict.py`

Stale or broken API-related path:

- `api/inference.py` loads `model_artifacts/fraud_lgbm_v1.pkl`, but the tracked model artifact is `fraud_lgbm_v1.joblib`.
- `api/inference.py` is not imported by `api/main.py` and is not used by the Docker entry points found.
- `api/inference.py` also computes `PROJECT_ROOT = Path(__file__).resolve().parents[1]`, which resolves to the repository root from `api/inference.py`; it then appends `model_artifacts`, so the artifact directory itself is plausible, but the file extension is not.

API limitations:

- Input schema is a generic `dict`; no strict transaction schema is enforced.
- No API authentication or authorization was found.
- No rate limiting was found.
- Errors are collapsed to `500 Prediction error`, which is safe from leaking internals but weak for operational diagnostics.

## Data and lakehouse flow

Lakehouse directories and evidence:

- `lakehouse/raw/`: train CSV and Parquet files.
- `lakehouse/external/`: test CSV and Parquet files.
- `lakehouse/processed/`: processed training/test Parquet files.
- `lakehouse/curated/`: curated training/test/prediction Parquet files.
- `lakehouse/splits/`: train/validation splits for training.
- `lakehouse/transformations/`: batch processing scripts.

Implemented batch scripts:

- `lakehouse/transformations/process_test_batch.py`
- `lakehouse/transformations/transform_test_batch.py`
- `lakehouse/transformations/run_batch_prediction.py`
- `scripts/build_curated_dataset.py`

Broken or risky data script:

- `scripts/ingest_raw_to_parquet.py` sets `BASE_DIR = Path(__file__).resolve().parent[1]`. `Path.parent` is a `Path`, not an indexable parent sequence. This path expression is expected to fail if executed.

Reproducibility gap:

- Run manifests reference old absolute paths under `D:\Dev\enterprise-aws-data-lakehouse-ml-system\...`, not the current repository path.
- Training manifest contains the hash of `X_train.parquet`, but not hashes for all splits, model, transformer, feature columns, threshold, or metadata.

## Streaming and orchestration

Kafka streaming:

| Component | Status | Evidence |
|---|---:|---|
| Kafka broker in Docker Compose | Implemented | `docker-compose.yml` defines single-node KRaft Kafka |
| API service in Docker Compose | Implemented | builds `api/Dockerfile`, exposes `8000` |
| Kafka consumer | Implemented / basic | `api/streaming/consumer.py` consumes events and posts to `/predict` |
| Kafka producer | Simulated | `api/streaming/producer.py` sends one sample transaction |
| Result topic / durable output | Unsupported | no evidence found |
| DLQ / poison-message handling | Unsupported | no evidence found |
| Event schema validation | Unsupported | no evidence found |

Airflow:

| DAG | Status | Evidence |
|---|---:|---|
| `batch_scoring_dag.py` | Implemented / unvalidated | Chains process, transform, and predict scripts |
| `retrain_pipeline.py` | Stale or broken | Calls missing `ml/registry/register_model.py`; monitoring scripts are function modules, not runnable full tasks |
| Airflow compose | Simulated / local | `orchestration/airflow/docker-compose.airflow.yml` defines local Airflow and Postgres |

## Cloud and deployment

Docker:

- `api/Dockerfile` is the API image used by root `docker-compose.yml`.
- `docker/Dockerfile` is the image built by GitHub Actions CI.
- `api/streaming/Dockerfile` packages the Kafka consumer.

AWS / Terraform:

Implemented Terraform modules:

- `terraform/modules/vpc`
- `terraform/modules/s3`
- `terraform/modules/iam`
- `terraform/modules/alb`
- `terraform/modules/ecs`
- `terraform/modules/ecr`

Dev environment:

- `terraform/environments/dev/main.tf` wires VPC, S3, IAM, ALB, and ECS.
- `terraform/environments/dev/terraform.tfvars` contains an account-specific ECR image URI and deployment values.

Cloud status:

- Terraform code exists.
- No Terraform plan/apply was run during this audit.
- No Terraform state, deployment output, live ALB health check, or current AWS endpoint validation was verified.
- Current deployment should be classified as planned/unvalidated, not production-ready.

Terraform inconsistency:

- An ECR module exists but is not wired into `terraform/environments/dev/main.tf`.
- ECS consumes an externally supplied image URI rather than an in-repo ECR module output.

## Testing and CI/CD

Tests found:

- `tests/test_api.py`
- `tests/test_inference.py`
- `tests/test_model_artifacts.py`
- `tests/test_training.py`
- `tests/test_data_pipeline.py`

CI found:

- `.github/workflows/ci.yml`
- Runs on push or pull request to `main`.
- Installs `requirements.txt`.
- Runs `python -m pytest tests/`.
- Builds Docker image from `docker/Dockerfile`.

Validation classification:

- Basic test scaffolding is implemented.
- CI is implemented for tests and image build only.
- CI/CD is not a full deployment pipeline.
- No coverage gate, type check, lint step, security scan, Terraform validation, Docker Compose integration test, Kafka integration test, Airflow validation, or deployment stage was found.

Tests were not executed during this audit because:

- This was a read-only audit.
- `tests/test_api.py` imports the live app and can append to `artifacts/metrics/api_metrics.jsonl` through `/predict`.
- Running tests could therefore modify repository artifacts, violating the requested preservation scope.

## Monitoring and governance

Implemented monitoring evidence:

- `api/main.py` exposes `/metrics` with Prometheus `Counter` and `Histogram`.
- `artifacts/metrics/metrics_file_logger.py` appends API prediction events to `artifacts/metrics/api_metrics.jsonl`.
- `monitoring/prometheus/prometheus.yml` exists.
- `ml/monitoring/data_drift.py` implements PSI helpers.
- `ml/monitoring/model_metrics.py` implements metric helpers.
- `ml/monitoring/prediction_monitor.py` tracks in-memory prediction tier counts.
- `observability/health.py`, `observability/logging_config.py`, `observability/tracing.py`, and `observability/metrics.py` exist.

Monitoring limitations:

- Monitoring modules are mostly not wired into the active API path, except direct Prometheus metrics in `api/main.py` and JSONL metric logging.
- No alerting rules were found.
- No dashboard provisioning was found.
- No production retention, centralized logging, or incident response workflow was found.
- `observability/metrics.py` defines metric names that differ from `api/main.py`, suggesting duplicate/unwired observability code.

Governance evidence:

- `AI_USAGE.md` exists and states AI assistance was reviewed by the maintainer.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`, pull request template, and issue templates exist.

## Active vs stale component table

| Component | Classification | Evidence / reason |
|---|---:|---|
| `api/main.py` | Active | Docker, tests, and imports point here |
| `ml/inference/predict.py` | Active | Used by `api/main.py` |
| `model_artifacts/*_v1` | Active | Loaded by active predictor |
| `api/inference.py` | Stale or broken | Expects missing `.pkl`; not active import path |
| `ml/pipelines/inference_pipeline.py` | Implemented but secondary | Similar inference flow, not active API path |
| `ml/pipelines/streaming_inference_pipeline.py` | Simulated | Sample local scorer; not wired into Docker Compose consumer |
| `api/streaming/consumer.py` | Implemented basic | Docker Compose consumer posts to FastAPI |
| `api/streaming/producer.py` | Simulated | Single sample producer |
| `orchestration/airflow/dags/batch_scoring_dag.py` | Implemented / unvalidated | BashOperator chain exists |
| `orchestration/airflow/dags/retrain_pipeline.py` | Stale or broken | References missing registry script |
| `ml/registry/model_registry.py` | Unsupported / empty | File exists but has no implementation |
| `ml/registry/versioning.py` | Unsupported / empty | File exists but has no implementation |
| `scripts/ingest_raw_to_parquet.py` | Broken | Invalid `Path.parent[1]` expression |
| `configs/api_config.yaml` | Stale or unused | Points to `artifacts/models/...`, but active runtime uses `model_artifacts/...` directly |
| `configs/model_config.yaml` | Planned / unused | Hyperparameters duplicated in code |
| `configs/pipeline_config.yaml` | Planned / partly aligned | Values mirror code but are not loaded by active training path |
| Terraform modules | Implemented / unvalidated | IaC exists, no current deploy evidence |
| GitHub Actions CI | Implemented | Tests and Docker build only |
| Notebooks | Historical evidence | Useful baseline exploration, not active runtime |

## Capability status matrix

| Capability | Status | Evidence |
|---|---:|---|
| FastAPI inference API | Implemented | `api/main.py` |
| API artifact loading | Validated by inspection | model, transformer, threshold, feature columns load successfully |
| Strict API input schema | Unsupported | request model is `data: dict` |
| LightGBM model serving | Implemented | active predictor loads LightGBM joblib |
| Feature transformer reuse | Implemented | active predictor loads serialized transformer |
| Feature contract | Validated by inspection, with risk | 445-column contract verified; source/artifact mismatch risk |
| Threshold loading | Implemented | `threshold_v1.json` loaded by active predictor |
| Training pipeline | Implemented | `ml/pipelines/training_pipeline.py` |
| Training reproducibility | Partially validated | manifest exists, but absolute old paths and incomplete hashes |
| Evaluation metrics | Implemented | `ml/training/evaluate.py` and manifest metrics |
| Batch scoring | Implemented / validated by artifacts | scripts and batch manifest/prediction file exist |
| Lakehouse organization | Implemented | raw, external, processed, curated, splits directories exist |
| Kafka event-to-API scoring | Implemented / basic | Docker Compose and consumer exist |
| Production streaming platform | Unsupported | no schema registry, DLQ, result topic, replay controls, or integration evidence |
| Airflow batch scoring | Implemented / unvalidated | DAG exists |
| Airflow retraining | Stale or broken | missing registry script |
| Docker API packaging | Implemented | API and general Dockerfiles |
| AWS infrastructure code | Implemented / unvalidated | Terraform modules and dev env exist |
| Current AWS deployment | Unsupported | no live validation performed or evidence found |
| CI tests | Implemented | GitHub Actions workflow |
| CD deployment | Unsupported | no deployment job found |
| Prometheus API metrics | Implemented | `/metrics` mounted in `api/main.py` |
| Drift monitoring | Planned / library only | helper code exists, not wired into runtime |
| Model registry | Unsupported | empty registry files; missing script referenced by Airflow |
| Explainability | Planned / partial | SHAP files exist but not inspected as active runtime |
| Agentic AI | Planned | no runtime evidence found |
| Selective inference | Planned | no runtime evidence found |

## Known risks and unsupported claims

High-priority risks:

1. Active API threshold governance is unresolved. Runtime uses `0.008540712517184246`, while existing report evidence notes notebook/business-threshold alternatives. Do not claim one official business threshold without a controlled decision.
2. Serialized feature transformer and current feature engineering source may not be fully reproducible because active artifacts include `uid_time_to_next` and `uid_time_from_prev`, while current source does not visibly create them.
3. `api/inference.py` is stale or broken and can confuse maintainers about the active inference path.
4. Retraining orchestration is broken because `retrain_pipeline.py` references missing `ml/registry/register_model.py`.
5. Config files do not drive the active runtime. Some paths in `configs/api_config.yaml` do not match active artifact paths.
6. API lacks authentication, authorization, strict request validation, and rate limiting.
7. Terraform deployment is unvalidated in this audit and should not be represented as currently deployed or production-ready.
8. Tests are smoke-level and do not prove production behavior, data contracts, Kafka integration, Airflow execution, Terraform validity, or API reliability.

Unsupported or overclaim-prone areas:

- Fully production-ready deployment
- Current live AWS deployment
- Full CI/CD
- Continuous retraining
- Complete model registry
- Agentic AI decisioning
- Selective inference
- Production-grade streaming
- End-to-end observability and alerting
- Calibrated business operating threshold
- High availability, HTTPS, autoscaling validation, or rollback execution

## Recommended target architecture

Safest rebuild target:

```text
1. Single source of truth for model artifacts
   model_artifacts/
   manifest with hashes for model, transformer, feature columns, threshold, metadata, splits

2. Single inference package
   ml/inference/predict.py
   remove or quarantine stale duplicate paths later after review

3. Explicit schema contract
   Pydantic request schema
   feature input schema
   artifact feature schema check

4. Reproducible training pipeline
   config-driven training
   deterministic split inputs
   complete run manifest
   artifact hash manifest

5. Batch and streaming as separate layers
   batch: lakehouse transformations and batch scoring
   streaming: Kafka consumer, schema validation, result topic, DLQ

6. Deployment layers
   local Docker Compose first
   CI validation second
   Terraform plan validation third
   AWS deployment only after artifact/runtime contract is stable

7. Monitoring
   API metrics
   model prediction metrics
   drift reports
   alert rules
   dashboard docs
```

Do not implement agentic AI or selective inference until the baseline artifact contract, threshold decision, API schema, and reproducibility story are stable.

## Recommended README structure

Recommended README structure based only on verified evidence:

1. Project overview
   - Say "production-oriented fraud detection ML system."
   - Do not say "production-ready."

2. Current verified capabilities
   - FastAPI inference
   - LightGBM artifact serving
   - lakehouse-style local data layout
   - batch scoring scripts
   - basic Kafka consumer-to-API flow
   - Airflow DAG definitions
   - Docker packaging
   - Terraform scaffolding
   - CI smoke tests

3. Architecture diagram
   - Separate active API path from planned/stale components.

4. Active runtime path
   - `api.main` and `ml.inference.predict.FraudPredictor`.

5. Model artifacts and feature contract
   - List the five active `model_artifacts/*_v1` files.
   - State current threshold and feature count as artifact facts, not business approval.

6. ML lifecycle
   - Training, thresholding, evaluation, batch scoring.

7. Data lakehouse layout
   - raw, external, processed, curated, splits.

8. Local development
   - install dependencies, run API, run tests with caution if metrics file is tracked.

9. Docker and streaming
   - local Compose API/Kafka/consumer flow.

10. Cloud/IaC
   - Terraform scaffolding only; no current deployment claim unless separately verified.

11. Testing and validation status
   - Be explicit about smoke tests vs missing integration/production validation.

12. Known limitations
   - stale paths, missing registry, threshold governance, auth, monitoring gaps.

13. AI usage and governance
   - Link `AI_USAGE.md`.

## Safe next implementation sequence

1. Decide and document the official baseline/threshold objective without changing artifacts yet.
2. Add a small runtime architecture note identifying `api.main` as active and `api/inference.py` as stale, or remove/quarantine stale duplicate code in a later controlled change.
3. Add artifact manifest hashes for model, transformer, feature columns, threshold, metadata, and split files.
4. Reconcile current feature engineering source with the serialized 445-feature artifact contract.
5. Introduce strict Pydantic request schemas for the active API.
6. Add non-mutating tests for artifact loading and schema validation.
7. Add Docker Compose integration validation for API health and prediction.
8. Fix or disable broken retraining DAG references.
9. Wire configs into training/inference only after the active runtime contract is stable.
10. Add Terraform validation in CI before claiming deployability.
11. Add streaming schema validation, result topic, and DLQ before claiming production streaming.
12. Revisit selective inference only after baseline, schema, and monitoring are validated.

## Files inspected

Source and runtime:

- `api/main.py`
- `api/inference.py`
- `ml/inference/predict.py`
- `ml/pipelines/inference_pipeline.py`
- `ml/pipelines/streaming_inference_pipeline.py`
- `ml/pipelines/training_pipeline.py`
- `ml/training/feature_engineering.py`
- `ml/training/train_lgbm.py`
- `ml/training/evaluate.py`
- `ml/utils/threshold.py`
- `ml/utils/run_manifest.py`

Data and batch:

- `scripts/ingest_raw_to_parquet.py`
- `scripts/build_curated_dataset.py`
- `lakehouse/transformations/process_test_batch.py`
- `lakehouse/transformations/transform_test_batch.py`
- `lakehouse/transformations/run_batch_prediction.py`

Streaming and orchestration:

- `api/streaming/consumer.py`
- `api/streaming/producer.py`
- `api/streaming/Dockerfile`
- `orchestration/airflow/dags/batch_scoring_dag.py`
- `orchestration/airflow/dags/retrain_pipeline.py`
- `orchestration/airflow/docker-compose.airflow.yml`

Deployment and infrastructure:

- `api/Dockerfile`
- `docker/Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `terraform/environments/dev/main.tf`
- `terraform/environments/dev/terraform.tfvars`
- `terraform/modules/vpc/main.tf`
- `terraform/modules/s3/main.tf`
- `terraform/modules/iam/main.tf`
- `terraform/modules/alb/main.tf`
- `terraform/modules/ecs/main.tf`
- `terraform/modules/ecr/main.tf`

Config, monitoring, governance, tests, docs:

- `configs/api_config.yaml`
- `configs/model_config.yaml`
- `configs/pipeline_config.yaml`
- `configs/data_config.yaml`
- `configs/config_loader.py`
- `ml/monitoring/data_drift.py`
- `ml/monitoring/model_metrics.py`
- `ml/monitoring/prediction_monitor.py`
- `observability/health.py`
- `observability/logging_config.py`
- `observability/metrics.py`
- `observability/tracing.py`
- `artifacts/metrics/metrics_file_logger.py`
- `tests/test_api.py`
- `tests/test_inference.py`
- `tests/test_model_artifacts.py`
- `tests/test_training.py`
- `tests/test_data_pipeline.py`
- `docs/api.md`
- `docs/architecture.md`
- `docs/data_lakehouse.md`
- `docs/ml_pipeline.md`
- `docs/reports/current_project_repository_audit.md`
- `docs/reports/lightgbm_baseline_reconciliation.md`
- `AI_USAGE.md`
- `requirements.txt`
- `pytest.ini`

Artifacts and manifests:

- `model_artifacts/metadata_v1.json`
- `model_artifacts/threshold_v1.json`
- `model_artifacts/feature_columns_v1.json`
- `model_artifacts/fraud_lgbm_v1.joblib`
- `model_artifacts/feature_transformer_v1.joblib`
- `artifacts/runs/training_20260304T155620Z/manifest.json`
- `artifacts/runs/batch_20260304T155733Z/manifest.json`

## Commands run

Read-only inspection commands:

```powershell
git status --short --branch
Get-ChildItem -Force
rg --files
Get-Content api\main.py
Get-Content api\inference.py
Get-Content ml\inference\predict.py
Get-Content docker-compose.yml
Get-Content ml\training\train_lgbm.py
Get-Content ml\training\feature_engineering.py
Get-Content ml\training\evaluate.py
Get-Content ml\pipelines\training_pipeline.py
Get-Content ml\pipelines\inference_pipeline.py
Get-Content ml\pipelines\streaming_inference_pipeline.py
Get-Content ml\utils\threshold.py
Get-Content ml\utils\run_manifest.py
Get-ChildItem .github -Recurse -Force
Get-Content tests\test_api.py
Get-Content tests\test_inference.py
Get-Content tests\test_model_artifacts.py
Get-Content tests\test_training.py
Get-Content tests\test_data_pipeline.py
Get-Content .github\workflows\ci.yml
Get-Content api\Dockerfile
Get-Content docker\Dockerfile
Get-Content api\streaming\consumer.py
Get-Content api\streaming\producer.py
Get-Content api\streaming\Dockerfile
Get-Content lakehouse\transformations\process_test_batch.py
Get-Content lakehouse\transformations\transform_test_batch.py
Get-Content lakehouse\transformations\run_batch_prediction.py
Get-Content scripts\ingest_raw_to_parquet.py
Get-Content scripts\build_curated_dataset.py
Get-Content orchestration\airflow\dags\batch_scoring_dag.py
Get-Content orchestration\airflow\dags\retrain_pipeline.py
Get-Content orchestration\airflow\docker-compose.airflow.yml
Get-Content ml\monitoring\data_drift.py
Get-Content ml\monitoring\model_metrics.py
Get-Content ml\monitoring\prediction_monitor.py
Get-Content observability\metrics.py
Get-Content observability\health.py
Get-Content observability\logging_config.py
Get-Content observability\tracing.py
Get-Content artifacts\metrics\metrics_file_logger.py
Get-Content configs\config_loader.py
Get-Content configs\api_config.yaml
Get-Content configs\model_config.yaml
Get-Content configs\pipeline_config.yaml
Get-Content configs\data_config.yaml
Get-Content model_artifacts\metadata_v1.json
Get-Content model_artifacts\threshold_v1.json
Get-Content model_artifacts\feature_columns_v1.json
Get-Content terraform\environments\dev\main.tf
Get-Content terraform\modules\ecs\main.tf
Get-Content terraform\modules\s3\main.tf
Get-Content terraform\modules\alb\main.tf
Get-Content terraform\modules\ecr\main.tf
Get-Content terraform\modules\iam\main.tf
Get-Content terraform\modules\vpc\main.tf
Get-Content terraform\environments\dev\terraform.tfvars
rg -n "agent|agentic|selective|fallback|human|threshold|FastAPI|uvicorn|api.main|api.inference|FraudPredictor|register_model|uid_time|production|deploy"
Get-ChildItem lakehouse -Recurse -Force
Get-ChildItem artifacts -Recurse -Force
Get-ChildItem model_artifacts -Force
Get-Content artifacts\runs\training_20260304T155620Z\manifest.json
Get-Content artifacts\runs\batch_20260304T155733Z\manifest.json
Get-Content requirements.txt
Get-Content pytest.ini
```

Read-only artifact inspection command:

```powershell
@'
from pathlib import Path
import joblib, json
root = Path.cwd()
trans = joblib.load(root / 'model_artifacts' / 'feature_transformer_v1.joblib')
model = joblib.load(root / 'model_artifacts' / 'fraud_lgbm_v1.joblib')
cols = json.loads((root / 'model_artifacts' / 'feature_columns_v1.json').read_text())
print('transformer_class=', type(trans).__module__ + '.' + type(trans).__name__)
print('transformer_schema_len=', len(getattr(trans, 'feature_schema', [])))
print('transformer_categorical_len=', len(getattr(trans, 'categorical_cols', [])))
print('feature_columns_len=', len(cols))
print('schema_equals_feature_columns=', getattr(trans, 'feature_schema', []) == cols)
print('model_class=', type(model).__module__ + '.' + type(model).__name__)
print('model_n_features=', getattr(model, 'n_features_in_', None))
print('model_best_iteration=', getattr(model, 'best_iteration_', None))
print('first_10_features=', cols[:10])
print('last_15_features=', cols[-15:])
'@ | bash -lc '.venv/bin/python -'
```

## Commands not run and why

- `pytest`: not run because the audit was read-only and API tests can append to `artifacts/metrics/api_metrics.jsonl`.
- `python -m pytest tests/`: not run for the same reason.
- `uvicorn api.main:app`: not run because starting the API was outside the read-only audit scope and could write metrics/log artifacts.
- `docker compose up`: not run because it starts services, creates containers/volumes, and is outside the requested preservation scope.
- `docker build`: not run because it creates local Docker artifacts.
- Airflow commands: not run because they start services or execute pipelines.
- Training, evaluation, batch scoring, and lakehouse transformation scripts: not run because they can overwrite artifacts, manifests, or Parquet outputs.
- Terraform `init`, `validate`, `plan`, or `apply`: not run because cloud/IaC execution and generated files were outside the audit scope.
- Any command reading `.env`: not run to avoid exposing secrets or local-only configuration.
- Any git branch, tag, commit, push, or remote command: not run because the task explicitly prohibited these changes.
