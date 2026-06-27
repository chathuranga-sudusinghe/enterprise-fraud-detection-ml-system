import json
import pathlib
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ARTIFACT_DIR = Path("model_artifacts")
EXPECTED_FEATURE_COUNT = 831
EXPECTED_MODEL_VERSION = "v2"
EXPECTED_MODEL_FAMILY = "catboost"
EXPECTED_THRESHOLD = 0.10


class FraudPredictorV2:
    def __init__(self, artifact_dir: str | Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.model: Any | None = None
        self.feature_engine: Any | None = None
        self.feature_columns: list[str] | None = None
        self.metadata: dict[str, Any] | None = None
        self.threshold: float | None = None
        self._loaded = False

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded()

        if self.feature_engine is None or self.model is None:
            raise RuntimeError("Model v2 artifacts are not loaded")
        if self.feature_columns is None or self.threshold is None:
            raise RuntimeError("Model v2 metadata is not loaded")

        transformed = self.feature_engine.transform(input_df)
        missing_features = [
            feature for feature in self.feature_columns if feature not in transformed.columns
        ]
        if missing_features:
            raise ValueError(
                "Model v2 transformed features are missing required columns: "
                f"{missing_features[:10]}"
            )

        transformed = transformed.reindex(columns=self.feature_columns)
        if transformed.isnull().any().any():
            raise ValueError("Model v2 transformed features contain null values")
        if transformed.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Model v2 requires {EXPECTED_FEATURE_COUNT} features; "
                f"found {transformed.shape[1]}"
            )

        y_proba = self.model.predict_proba(transformed)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        result = input_df.copy()
        result["fraud_probability"] = y_proba
        result["fraud_prediction"] = y_pred
        result["threshold"] = self.threshold
        result["model_version"] = EXPECTED_MODEL_VERSION
        result["model_family"] = EXPECTED_MODEL_FAMILY
        result["feature_count"] = EXPECTED_FEATURE_COUNT
        return result

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        artifact_paths = {
            "model": self.artifact_dir / "fraud_catboost_v2.joblib",
            "transformer": self.artifact_dir / "feature_transformer_v2.joblib",
            "feature_columns": self.artifact_dir / "feature_columns_v2.json",
            "metadata": self.artifact_dir / "metadata_v2.json",
            "threshold": self.artifact_dir / "threshold_v2.json",
        }
        missing_paths = [
            str(path) for path in artifact_paths.values() if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing Model v2 artifact files: {missing_paths}"
            )

        self.model = _load_joblib_path(artifact_paths["model"])
        self.feature_engine = _load_joblib_path(artifact_paths["transformer"])
        self.feature_columns = _load_json(artifact_paths["feature_columns"])
        self.metadata = _load_json(artifact_paths["metadata"])
        threshold_payload = _load_json(artifact_paths["threshold"])

        self._validate_loaded_artifacts(threshold_payload)
        self._loaded = True

    def _validate_loaded_artifacts(self, threshold_payload: Any) -> None:
        if not isinstance(self.feature_columns, list):
            raise ValueError("feature_columns_v2.json must contain a list")
        if len(self.feature_columns) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Model v2 feature_columns_v2.json must contain "
                f"{EXPECTED_FEATURE_COUNT} features; found {len(self.feature_columns)}"
            )

        if not isinstance(self.metadata, dict):
            raise ValueError("metadata_v2.json must contain an object")
        if self.metadata.get("model_version") != EXPECTED_MODEL_VERSION:
            raise ValueError("metadata_v2.json model_version must be v2")
        if self.metadata.get("model_family") != EXPECTED_MODEL_FAMILY:
            raise ValueError("metadata_v2.json model_family must be catboost")
        if self.metadata.get("n_features") != EXPECTED_FEATURE_COUNT:
            raise ValueError("metadata_v2.json n_features must be 831")

        if not isinstance(threshold_payload, dict):
            raise ValueError("threshold_v2.json must contain an object")
        threshold = threshold_payload.get("threshold")
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold_v2.json threshold must be numeric")
        self.threshold = float(threshold)
        if abs(self.threshold - EXPECTED_THRESHOLD) > 1e-12:
            raise ValueError("threshold_v2.json threshold must be 0.10")
        if abs(float(self.metadata.get("threshold")) - EXPECTED_THRESHOLD) > 1e-12:
            raise ValueError("metadata_v2.json threshold must be 0.10")


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def _load_joblib_path(path: Path) -> Any:
    if sys.platform != "win32":
        return joblib.load(path)

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return joblib.load(path)
    finally:
        pathlib.PosixPath = original_posix_path
