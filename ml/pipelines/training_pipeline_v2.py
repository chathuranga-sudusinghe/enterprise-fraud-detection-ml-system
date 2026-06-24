from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.training.feature_engineering_v2 import FeatureEngineeringV2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_V2_PATH = ARTIFACT_DIR / "fraud_lgbm_v2.joblib"
TRANSFORMER_V2_PATH = ARTIFACT_DIR / "feature_transformer_v2.joblib"
FEATURE_COLUMNS_V2_PATH = ARTIFACT_DIR / "feature_columns_v2.json"
METADATA_V2_PATH = ARTIFACT_DIR / "metadata_v2.json"
THRESHOLD_V2_PATH = ARTIFACT_DIR / "threshold_v2.json"

V2_ARTIFACT_PATHS = {
    "model": MODEL_V2_PATH,
    "transformer": TRANSFORMER_V2_PATH,
    "feature_columns": FEATURE_COLUMNS_V2_PATH,
    "metadata": METADATA_V2_PATH,
    "threshold": THRESHOLD_V2_PATH,
}

KNOWN_V1_ARTIFACT_NAMES = {
    "fraud_lgbm_v1.joblib",
    "feature_transformer_v1.joblib",
    "feature_columns_v1.json",
    "metadata_v1.json",
    "threshold_v1.json",
}


def validate_v2_artifact_paths(artifact_paths: dict[str, Path]) -> None:
    """Prevent v2 pipeline code from targeting persisted v1 artifact files."""

    for name, path in artifact_paths.items():
        artifact_path = Path(path)
        artifact_name = artifact_path.name
        artifact_stem = artifact_path.stem

        if artifact_name in KNOWN_V1_ARTIFACT_NAMES or artifact_stem.endswith("_v1"):
            raise ValueError(
                f"Unsafe v2 artifact path for {name}: {artifact_path}. "
                "V2 training must not write or target v1 artifacts."
            )


def run_training_pipeline_v2_dry_run(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    *,
    artifact_paths: dict[str, Path] | None = None,
    transformer: FeatureEngineeringV2 | None = None,
) -> dict[str, Any]:
    """
    Exercise the v2 feature pipeline without training or writing artifacts.

    This function is intentionally in-memory and dry-run only. It exists to make
    the future v2 training pipeline testable before model training is enabled.
    """

    selected_artifact_paths = artifact_paths or V2_ARTIFACT_PATHS
    validate_v2_artifact_paths(selected_artifact_paths)
    _validate_xy_lengths(X_train, y_train, "train")
    _validate_xy_lengths(X_val, y_val, "val")

    if X_test is not None and y_test is not None:
        _validate_xy_lengths(X_test, y_test, "test")

    feature_transformer = transformer or FeatureEngineeringV2()
    X_train_v2 = feature_transformer.fit_transform(X_train)
    X_val_v2 = feature_transformer.transform(X_val)
    X_test_v2 = feature_transformer.transform(X_test) if X_test is not None else None

    summary: dict[str, Any] = {
        "feature_engineering_version": "v2",
        "transformer_class": feature_transformer.__class__.__name__,
        "train_shape": tuple(X_train_v2.shape),
        "val_shape": tuple(X_val_v2.shape),
        "feature_count": len(feature_transformer.feature_names_),
        "feature_names": list(feature_transformer.feature_names_),
        "artifact_paths": {
            name: str(path) for name, path in sorted(selected_artifact_paths.items())
        },
        "would_write_artifacts": False,
    }

    if X_test_v2 is not None:
        summary["test_shape"] = tuple(X_test_v2.shape)

    return summary


def _validate_xy_lengths(X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    if len(X) != len(y):
        raise ValueError(
            f"X_{split_name} and y_{split_name} length mismatch: "
            f"{len(X)} != {len(y)}"
        )
