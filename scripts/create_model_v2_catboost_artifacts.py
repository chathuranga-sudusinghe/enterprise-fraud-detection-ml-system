from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.experiments.model_v2_catboost_tuning import (  # noqa: E402
    _build_catboost_pool,
    _catboost_params,
    get_catboost_availability,
)
from ml.pipelines.training_pipeline_v2 import (  # noqa: E402
    ARTIFACT_DIR,
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TRANSACTION_PATH,
    KNOWN_V1_ARTIFACT_NAMES,
    align_categorical_features_for_lightgbm,
    evaluate_predictions_v2,
    load_transaction_identity_dataset,
    prepare_time_based_train_val_test_split,
    validate_feature_columns_match,
    validate_v2_artifact_paths,
)
from ml.training.feature_engineering_v2 import FeatureEngineeringV2  # noqa: E402


EXPECTED_MODEL_V2_FEATURE_COUNT = 831
CATBOOST_V2_THRESHOLD = 0.10

CATBOOST_V2_ARTIFACT_PATHS = {
    "model": ARTIFACT_DIR / "fraud_catboost_v2.joblib",
    "transformer": ARTIFACT_DIR / "feature_transformer_v2.joblib",
    "feature_columns": ARTIFACT_DIR / "feature_columns_v2.json",
    "metadata": ARTIFACT_DIR / "metadata_v2.json",
    "threshold": ARTIFACT_DIR / "threshold_v2.json",
    "evaluation_report": ARTIFACT_DIR / "model_v2_evaluation_report.json",
}


def create_model_v2_catboost_artifacts(
    *,
    transaction_path: str | Path = DEFAULT_TRANSACTION_PATH,
    identity_path: str | Path = DEFAULT_IDENTITY_PATH,
    artifact_paths: dict[str, Path] | None = None,
    write_artifacts: bool = False,
    overwrite_v2: bool = False,
    expected_feature_count: int = EXPECTED_MODEL_V2_FEATURE_COUNT,
    threshold: float = CATBOOST_V2_THRESHOLD,
) -> dict[str, Any]:
    """
    Train the validated CatBoost Model v2 candidate and optionally write artifacts.

    Artifact writing is disabled by default. When enabled, only v2-specific
    artifact paths are allowed, existing v2 artifacts require ``overwrite_v2``,
    and v1 artifact names are rejected.
    """

    selected_artifact_paths = artifact_paths or CATBOOST_V2_ARTIFACT_PATHS
    validate_catboost_v2_artifact_paths(
        selected_artifact_paths,
        write_artifacts=write_artifacts,
        overwrite_v2=overwrite_v2,
    )
    if abs(threshold - CATBOOST_V2_THRESHOLD) > 1e-12:
        raise ValueError(
            f"Model v2 CatBoost artifact creation requires threshold "
            f"{CATBOOST_V2_THRESHOLD:.2f}; received {threshold:.2f}."
        )

    merged = load_transaction_identity_dataset(
        transaction_path=transaction_path,
        identity_path=identity_path,
    )
    splits = prepare_time_based_train_val_test_split(merged)

    transformer = FeatureEngineeringV2()
    X_train_v2 = transformer.fit_transform(splits["X_train"])
    X_val_v2 = transformer.transform(splits["X_val"])
    X_test_v2 = transformer.transform(splits["X_test"])
    validate_feature_columns_match(X_train_v2, X_val_v2, X_test_v2)

    feature_count = len(transformer.feature_names_)
    if feature_count != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} Model v2 features, "
            f"but FeatureEngineeringV2 produced {feature_count}."
        )

    categorical_cols = [
        col for col in transformer.categorical_columns_ if col in X_train_v2.columns
    ]
    X_train_v2, X_val_v2, X_test_v2 = align_categorical_features_for_lightgbm(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
        categorical_cols=categorical_cols,
    )
    train_result = train_catboost_default_candidate(
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        X_test=X_test_v2,
        y_test=splits["y_test"],
        categorical_cols=categorical_cols,
    )

    validation_metrics = evaluate_predictions_v2(
        y_true=splits["y_val"],
        y_proba=train_result["val_proba"],
        threshold=threshold,
    )
    test_metrics = evaluate_predictions_v2(
        y_true=splits["y_test"],
        y_proba=train_result["test_proba"],
        threshold=threshold,
    )

    summary = {
        "model_version": "v2",
        "model_family": "catboost",
        "candidate": "catboost_default",
        "feature_engineering_version": "feature_engineering_v2",
        "transformer_class": transformer.__class__.__name__,
        "feature_count": feature_count,
        "categorical_feature_count": len(categorical_cols),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "execution_device": train_result["execution_device"],
        "catboost_params": train_result["params"],
        "artifact_paths": {
            name: str(path) for name, path in sorted(selected_artifact_paths.items())
        },
        "would_write_artifacts": write_artifacts,
        "overwrite_v2": overwrite_v2,
        "artifacts_written": False,
    }
    if train_result.get("gpu_error"):
        summary["gpu_error"] = train_result["gpu_error"]

    if write_artifacts:
        write_catboost_v2_artifacts(
            model=train_result["model"],
            transformer=transformer,
            summary=summary,
            artifact_paths=selected_artifact_paths,
        )
        summary["artifacts_written"] = True

    return summary


def validate_catboost_v2_artifact_paths(
    artifact_paths: dict[str, Path],
    *,
    write_artifacts: bool,
    overwrite_v2: bool,
) -> None:
    """Validate artifact paths before optional v2 artifact writing."""

    required_keys = set(CATBOOST_V2_ARTIFACT_PATHS)
    missing = sorted(required_keys - set(artifact_paths))
    if missing:
        raise ValueError(f"Missing required CatBoost v2 artifact paths: {missing}")

    validate_v2_artifact_paths(artifact_paths)
    for name, path in artifact_paths.items():
        artifact_path = Path(path)
        if artifact_path.name in KNOWN_V1_ARTIFACT_NAMES or artifact_path.stem.endswith("_v1"):
            raise ValueError(
                f"Unsafe CatBoost v2 artifact path for {name}: {artifact_path}"
            )
        if "_v2" not in artifact_path.stem:
            raise ValueError(
                f"CatBoost v2 artifact path for {name} must be v2-specific: "
                f"{artifact_path}"
            )

    if write_artifacts and not overwrite_v2:
        existing = [
            str(path)
            for path in artifact_paths.values()
            if Path(path).exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing v2 artifacts without "
                f"--overwrite-v2: {existing}"
            )


def train_catboost_default_candidate(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    """Train CatBoost default with GPU first and CPU fallback."""

    availability = get_catboost_availability()
    if not availability["available"]:
        raise RuntimeError(f"CatBoost is not available: {availability['reason']}")

    gpu_params = _catboost_params({}, task_type="GPU")
    try:
        return _fit_catboost_default_candidate(
            params=gpu_params,
            execution_device="GPU",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
        )
    except Exception as exc:
        cpu_params = _catboost_params({}, task_type="CPU")
        result = _fit_catboost_default_candidate(
            params=cpu_params,
            execution_device="CPU fallback",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
        )
        result["gpu_error"] = str(exc)
        return result


def _fit_catboost_default_candidate(
    *,
    params: dict[str, Any],
    execution_device: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    train_pool = _build_catboost_pool(X_train, y_train, categorical_cols)
    val_pool = _build_catboost_pool(X_val, y_val, categorical_cols)
    test_pool = _build_catboost_pool(X_test, y_test, categorical_cols)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return {
        "model": model,
        "execution_device": execution_device,
        "params": params,
        "val_proba": model.predict_proba(val_pool)[:, 1],
        "test_proba": model.predict_proba(test_pool)[:, 1],
    }


def write_catboost_v2_artifacts(
    *,
    model: Any,
    transformer: FeatureEngineeringV2,
    summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    """Write only validated v2 CatBoost artifacts."""

    for path in artifact_paths.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_paths["model"])
    joblib.dump(transformer, artifact_paths["transformer"])
    artifact_paths["feature_columns"].write_text(
        json.dumps(transformer.feature_names_, indent=2),
        encoding="utf-8",
    )
    artifact_paths["threshold"].write_text(
        json.dumps({"threshold": summary["threshold"]}, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "model_version": summary["model_version"],
        "model_family": summary["model_family"],
        "candidate": summary["candidate"],
        "feature_engineering_version": summary["feature_engineering_version"],
        "n_features": summary["feature_count"],
        "threshold": summary["threshold"],
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
        "execution_device": summary["execution_device"],
        "catboost_params": summary["catboost_params"],
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    artifact_paths["metadata"].write_text(
        json.dumps(to_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    evaluation_report = {
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
    }
    artifact_paths["evaluation_report"].write_text(
        json.dumps(to_json_safe(evaluation_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validated Model v2 CatBoost artifacts."
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Actually write v2 artifacts. Omit for dry-run training only.",
    )
    parser.add_argument(
        "--overwrite-v2",
        action="store_true",
        help="Allow replacing existing v2 artifact files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = create_model_v2_catboost_artifacts(
        write_artifacts=args.write_artifacts,
        overwrite_v2=args.overwrite_v2,
    )
    print(json.dumps(to_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
