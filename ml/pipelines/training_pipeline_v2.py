from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.training.feature_engineering_v2 import FeatureEngineeringV2
from ml.training.train_lgbm import train_lightgbm
from ml.utils.threshold import find_optimal_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "lakehouse" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"

DEFAULT_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.parquet"
DEFAULT_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.parquet"

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

TIME_COLUMN = "TransactionDT"
TARGET_COLUMN = "isFraud"
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
TRANSACTION_ID_COLUMN = "TransactionID"


def load_full_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load a full v2 training dataset from a supplied CSV or Parquet path."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported dataset file type for v2 training data: {suffix}. "
            "Expected .parquet or .csv."
        )


def load_transaction_identity_dataset(
    transaction_path: str | Path = DEFAULT_TRANSACTION_PATH,
    identity_path: str | Path = DEFAULT_IDENTITY_PATH,
) -> pd.DataFrame:
    """Load and left-merge IEEE transaction and identity training datasets."""

    transaction_df = load_full_dataset(transaction_path)
    identity_df = load_full_dataset(identity_path)
    return merge_transaction_identity(transaction_df, identity_df)


def merge_transaction_identity(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    *,
    id_column: str = TRANSACTION_ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Left-merge transaction rows with optional identity/device attributes.

    The transaction dataset is the row-preserving source of truth. Identity rows
    enrich matching transactions when available and remain missing for
    non-matching transaction IDs.
    """

    if id_column not in transaction_df.columns:
        raise ValueError(
            f"Missing merge key in transaction dataset: {id_column}"
        )
    if id_column not in identity_df.columns:
        raise ValueError(f"Missing merge key in identity dataset: {id_column}")
    if target_column not in transaction_df.columns:
        raise ValueError(
            f"Missing target column in transaction dataset: {target_column}"
        )
    if target_column in identity_df.columns:
        raise ValueError(
            f"Identity dataset must not contain target column: {target_column}"
        )

    return transaction_df.merge(identity_df, on=id_column, how="left")


def validate_time_split_required_columns(
    df: pd.DataFrame,
    *,
    time_column: str = TIME_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Validate columns required for deterministic time-based v2 splitting."""

    missing = [col for col in [time_column, target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required v2 split columns: {missing}")


def sort_by_transaction_time(
    df: pd.DataFrame,
    *,
    time_column: str = TIME_COLUMN,
) -> pd.DataFrame:
    """Return a stable TransactionDT-sorted copy without shuffling."""

    if time_column not in df.columns:
        raise ValueError(f"Missing required v2 split columns: ['{time_column}']")

    return df.sort_values(time_column, kind="mergesort").reset_index(drop=True)


def split_by_time_order(
    df: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    time_column: str = TIME_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split earliest rows to train, next rows to validation, latest rows to test."""

    validate_time_split_required_columns(
        df,
        time_column=time_column,
        target_column=target_column,
    )
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    sorted_df = sort_by_transaction_time(df, time_column=time_column)
    row_count = len(sorted_df)
    if row_count < 3:
        raise ValueError("Time-based v2 split requires at least 3 rows.")

    train_end = int(row_count * train_ratio)
    val_end = train_end + int(row_count * val_ratio)

    if train_end == 0 or val_end == train_end or val_end >= row_count:
        raise ValueError(
            "Time-based v2 split ratios produced an empty train, validation, "
            "or test split."
        )

    train_df = sorted_df.iloc[:train_end].reset_index(drop=True)
    val_df = sorted_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = sorted_df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def separate_target(
    df: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the fraud target column."""

    if target_column not in df.columns:
        raise ValueError(f"Missing target column for v2 training: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()
    return X, y


def prepare_time_based_train_val_test_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    time_column: str = TIME_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Prepare X/y train, validation, and test splits for v2 dry-run/training."""

    train_df, val_df, test_df = split_by_time_order(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        time_column=time_column,
        target_column=target_column,
    )
    X_train, y_train = separate_target(train_df, target_column=target_column)
    X_val, y_val = separate_target(val_df, target_column=target_column)
    X_test, y_test = separate_target(test_df, target_column=target_column)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
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


def run_training_pipeline_v2(
    *,
    transaction_path: str | Path = DEFAULT_TRANSACTION_PATH,
    identity_path: str | Path = DEFAULT_IDENTITY_PATH,
    artifact_paths: dict[str, Path] | None = None,
    write_artifacts: bool = False,
    target_recall: float = 0.95,
) -> dict[str, Any]:
    """
    Train the first LightGBM Model v2 using the safe v2 data flow.

    Artifacts are not written unless ``write_artifacts=True``. When writing is
    enabled, only explicit v2 artifact paths are allowed.
    """

    selected_artifact_paths = artifact_paths or V2_ARTIFACT_PATHS
    validate_v2_artifact_paths(selected_artifact_paths)

    merged = load_transaction_identity_dataset(
        transaction_path=transaction_path,
        identity_path=identity_path,
    )
    splits = prepare_time_based_train_val_test_split(merged)

    transformer = FeatureEngineeringV2()
    X_train_v2 = transformer.fit_transform(splits["X_train"])
    X_val_v2 = transformer.transform(splits["X_val"])
    X_test_v2 = transformer.transform(splits["X_test"])

    categorical_cols = [
        col for col in transformer.categorical_columns_ if col in X_train_v2.columns
    ]
    validate_feature_columns_match(X_train_v2, X_val_v2, X_test_v2)
    X_train_v2, X_val_v2, X_test_v2 = align_categorical_features_for_lightgbm(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
        categorical_cols=categorical_cols,
    )
    model, val_proba = train_lightgbm(
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        categorical_cols=categorical_cols,
    )
    test_proba = model.predict_proba(X_test_v2)[:, 1]

    threshold = float(
        find_optimal_threshold(
            y_true=splits["y_val"].to_numpy(),
            y_proba=np.asarray(val_proba),
            target_recall=target_recall,
        )
    )
    validation_metrics = evaluate_predictions_v2(
        y_true=splits["y_val"],
        y_proba=val_proba,
        threshold=threshold,
    )
    test_metrics = evaluate_predictions_v2(
        y_true=splits["y_test"],
        y_proba=test_proba,
        threshold=threshold,
    )

    summary: dict[str, Any] = {
        "feature_engineering_version": "v2",
        "model_version": "v2",
        "model_type": "lightgbm",
        "transformer_class": transformer.__class__.__name__,
        "train_shape": tuple(X_train_v2.shape),
        "val_shape": tuple(X_val_v2.shape),
        "test_shape": tuple(X_test_v2.shape),
        "feature_count": len(transformer.feature_names_),
        "feature_names": list(transformer.feature_names_),
        "numeric_feature_count": int(
            len(X_train_v2.select_dtypes(include=["number", "bool"]).columns)
        ),
        "categorical_feature_count": len(categorical_cols),
        "train_val_test_feature_columns_match": True,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "artifact_paths": {
            name: str(path) for name, path in sorted(selected_artifact_paths.items())
        },
        "would_write_artifacts": write_artifacts,
        "artifacts_written": False,
    }

    if write_artifacts:
        _write_v2_artifacts(
            model=model,
            transformer=transformer,
            threshold=threshold,
            summary=summary,
            artifact_paths=selected_artifact_paths,
        )
        summary["artifacts_written"] = True

    return summary


def validate_feature_columns_match(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Ensure train/validation/test model matrices have identical columns."""

    train_columns = list(X_train.columns)
    mismatches = {
        "validation": list(X_val.columns),
        "test": list(X_test.columns),
    }
    for split_name, columns in mismatches.items():
        if columns != train_columns:
            raise ValueError(
                f"V2 {split_name} feature columns do not match train columns."
            )


def align_categorical_features_for_lightgbm(
    *,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align categorical dtypes across train/validation/test before LightGBM.

    FeatureEngineeringV2 maps unknown validation/test categories to
    ``__UNKNOWN__``. This function gives all splits the same pandas
    CategoricalDtype so LightGBM receives matching categorical metadata.
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for col in categorical_cols:
        if col not in X_train.columns:
            continue
        categories = pd.Index(
            pd.concat(
                [
                    X_train[col].astype("string"),
                    X_val[col].astype("string"),
                    X_test[col].astype("string"),
                ],
                ignore_index=True,
            )
            .dropna()
            .unique()
        ).sort_values()
        dtype = CategoricalDtype(categories=categories.tolist())
        X_train[col] = X_train[col].astype("string").astype(dtype)
        X_val[col] = X_val[col].astype("string").astype(dtype)
        X_test[col] = X_test[col].astype("string").astype(dtype)

    return X_train, X_val, X_test


def evaluate_predictions_v2(
    y_true: pd.Series,
    y_proba: Any,
    threshold: float,
) -> dict[str, Any]:
    """Compute v2 binary-classification metrics including PR-AUC."""

    y_proba_array = np.asarray(y_proba)
    y_pred = (y_proba_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba_array)),
        "pr_auc": float(average_precision_score(y_true, y_proba_array)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "alert_rate": float((tp + fp) / len(y_true)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
    }


def _validate_xy_lengths(X: pd.DataFrame, y: pd.Series, split_name: str) -> None:
    if len(X) != len(y):
        raise ValueError(
            f"X_{split_name} and y_{split_name} length mismatch: "
            f"{len(X)} != {len(y)}"
        )


def _validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("V2 split ratios must all be positive.")

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("V2 split ratios must sum to 1.0.")


def _write_v2_artifacts(
    *,
    model: Any,
    transformer: FeatureEngineeringV2,
    threshold: float,
    summary: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> None:
    validate_v2_artifact_paths(artifact_paths)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_paths["model"])
    joblib.dump(transformer, artifact_paths["transformer"])
    artifact_paths["feature_columns"].write_text(
        json.dumps(transformer.feature_names_, indent=2),
        encoding="utf-8",
    )
    artifact_paths["threshold"].write_text(
        json.dumps({"threshold": threshold}, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_version": summary["model_version"],
        "feature_engineering_version": summary["feature_engineering_version"],
        "model_type": summary["model_type"],
        "n_features": summary["feature_count"],
        "threshold": threshold,
        "validation_metrics": summary["validation_metrics"],
        "test_metrics": summary["test_metrics"],
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    artifact_paths["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
