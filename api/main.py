from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import logging

# Prometheus monitoring
from prometheus_client import make_asgi_app, Counter, Histogram
import time

from ml.inference.predict import FraudPredictor
from ml.inference.predict_v2 import FraudPredictorV2
from ml.monitoring.metrics_file_logger import log_api_metric

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI(title="Fraud Detection API")


# -----------------------------
# Prometheus Metrics
# -----------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "model_version", "model_family", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint", "model_version", "model_family", "status"],
)

# expose /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# -----------------------------
# Load Model at Startup
# -----------------------------
predictor = FraudPredictor()
# Model v2 artifacts load lazily on first /predict/v2 request; /predict remains
# v1 and is not affected if v2 artifacts are missing.
predictor_v2 = FraudPredictorV2()


V1_METRIC_LABELS = {
    "endpoint": "/predict",
    "model_version": "v1",
    "model_family": "lightgbm",
}

V2_METRIC_LABELS = {
    "endpoint": "/predict/v2",
    "model_version": "v2",
    "model_family": "catboost",
}


# -----------------------------
# Input Schema
# -----------------------------
class TransactionInput(BaseModel):
    data: dict


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Fraud API is running"}


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# -----------------------------
# Prediction Endpoint v1
# -----------------------------
@app.post("/predict")
def predict(transaction: TransactionInput):
    start_time = time.time()

    try:
        logger.info("Received prediction request")

        df = pd.DataFrame([transaction.data])
        result = predictor.predict(df)

        response = {
            "fraud_probability": float(result["fraud_probability"].iloc[0]),
            "fraud_prediction": int(result["fraud_prediction"].iloc[0]),
        }

        # File-based metrics logging
        log_api_metric({
            **V1_METRIC_LABELS,
            "status": "success",
            "fraud_probability": response["fraud_probability"],
            "fraud_prediction": response["fraud_prediction"],
        })

        latency = time.time() - start_time
        REQUEST_COUNT.labels(**V1_METRIC_LABELS, status="success").inc()
        REQUEST_LATENCY.labels(**V1_METRIC_LABELS, status="success").observe(latency)

        logger.info("Prediction successful")
        return response

    except Exception as e:
        logger.exception("Prediction failed")
        latency = time.time() - start_time
        log_api_metric({
            **V1_METRIC_LABELS,
            "status": "error",
        })
        REQUEST_COUNT.labels(**V1_METRIC_LABELS, status="error").inc()
        REQUEST_LATENCY.labels(**V1_METRIC_LABELS, status="error").observe(latency)
        raise HTTPException(status_code=500, detail="Prediction error")


# -----------------------------
# Model v2 Prediction Endpoint
# -----------------------------
@app.post("/predict/v2")
def predict_v2(transaction: TransactionInput):
    start_time = time.time()

    try:
        logger.info("Received Model v2 prediction request")

        df = pd.DataFrame([transaction.data])
        result = predictor_v2.predict(df)

        response = {
            "fraud_probability": float(result["fraud_probability"].iloc[0]),
            "fraud_prediction": int(result["fraud_prediction"].iloc[0]),
            "threshold": float(result["threshold"].iloc[0]),
            "model_version": str(result["model_version"].iloc[0]),
            "model_family": str(result["model_family"].iloc[0]),
            "feature_count": int(result["feature_count"].iloc[0]),
        }

        log_api_metric({
            **V2_METRIC_LABELS,
            "status": "success",
            "fraud_probability": response["fraud_probability"],
            "fraud_prediction": response["fraud_prediction"],
            "threshold": response["threshold"],
            "feature_count": response["feature_count"],
        })

        latency = time.time() - start_time
        REQUEST_COUNT.labels(**V2_METRIC_LABELS, status="success").inc()
        REQUEST_LATENCY.labels(**V2_METRIC_LABELS, status="success").observe(latency)

        logger.info("Model v2 prediction successful")
        return response

    except Exception:
        logger.exception("Model v2 prediction failed")
        latency = time.time() - start_time
        log_api_metric({
            **V2_METRIC_LABELS,
            "status": "error",
        })
        REQUEST_COUNT.labels(**V2_METRIC_LABELS, status="error").inc()
        REQUEST_LATENCY.labels(**V2_METRIC_LABELS, status="error").observe(latency)
        raise HTTPException(status_code=500, detail="Model v2 prediction error")

