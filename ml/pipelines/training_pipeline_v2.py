from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.training.feature_engineering_v2 import FeatureEngineeringV2


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
