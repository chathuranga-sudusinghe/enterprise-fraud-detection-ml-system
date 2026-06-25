from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.training_pipeline_v2 import (  # noqa: E402
    TIME_COLUMN,
    align_categorical_features_for_lightgbm,
    load_transaction_identity_dataset,
    prepare_time_based_train_val_test_split,
    run_training_pipeline_v2_dry_run,
)
from ml.training.feature_engineering_v2 import FeatureEngineeringV2  # noqa: E402


def run_model_v2_data_flow_smoke_check() -> dict[str, Any]:
    """
    Run the v2 real-data flow without training or artifact writes.

    The flow is:
    raw transaction + identity parquet -> left merge -> time split ->
    FeatureEngineeringV2 dry-run summary.
    """

    merged = load_transaction_identity_dataset()
    splits = prepare_time_based_train_val_test_split(merged)
    dry_run_summary = run_training_pipeline_v2_dry_run(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
    )
    transformer = FeatureEngineeringV2()
    X_train_v2 = transformer.fit_transform(splits["X_train"])
    X_val_v2 = transformer.transform(splits["X_val"])
    X_test_v2 = transformer.transform(splits["X_test"])
    columns_match = (
        list(X_train_v2.columns) == list(X_val_v2.columns) == list(X_test_v2.columns)
    )
    categorical_cols = [
        col for col in transformer.categorical_columns_ if col in X_train_v2.columns
    ]
    aligned_train, aligned_val, aligned_test = align_categorical_features_for_lightgbm(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
        categorical_cols=categorical_cols,
    )
    category_alignment_warnings = _category_alignment_warnings(
        aligned_train,
        aligned_val,
        aligned_test,
        categorical_cols,
    )

    return {
        "merged_shape": tuple(merged.shape),
        "split_row_counts": {
            "train": int(len(splits["X_train"])),
            "validation": int(len(splits["X_val"])),
            "test": int(len(splits["X_test"])),
        },
        "transaction_dt_ranges": {
            "train": _min_max(splits["X_train"], TIME_COLUMN),
            "validation": _min_max(splits["X_val"], TIME_COLUMN),
            "test": _min_max(splits["X_test"], TIME_COLUMN),
        },
        "feature_count": int(dry_run_summary["feature_count"]),
        "numeric_feature_count": int(
            len(X_train_v2.select_dtypes(include=["number", "bool"]).columns)
        ),
        "categorical_feature_count": int(len(categorical_cols)),
        "train_val_test_feature_columns_match": columns_match,
        "would_write_artifacts": bool(dry_run_summary["would_write_artifacts"]),
        "transformer_class": dry_run_summary["transformer_class"],
        "warnings": category_alignment_warnings,
    }


def main() -> int:
    try:
        summary = run_model_v2_data_flow_smoke_check()
    except Exception as exc:
        print(f"Model v2 data flow smoke check failed: {exc}", file=sys.stderr)
        return 1

    print("Model v2 data flow smoke check summary:")
    pprint(summary)
    return 0


def _min_max(df: Any, column: str) -> dict[str, Any]:
    return {
        "min": _to_python_scalar(df[column].min()),
        "max": _to_python_scalar(df[column].max()),
    }


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _category_alignment_warnings(
    X_train: Any,
    X_val: Any,
    X_test: Any,
    categorical_cols: list[str],
) -> list[str]:
    warnings = []
    for col in categorical_cols:
        train_categories = list(X_train[col].cat.categories)
        if list(X_val[col].cat.categories) != train_categories:
            warnings.append(f"{col}: validation categories differ after alignment")
        if list(X_test[col].cat.categories) != train_categories:
            warnings.append(f"{col}: test categories differ after alignment")

    if not warnings:
        warnings.append("No categorical schema alignment warnings detected.")

    return warnings


if __name__ == "__main__":
    raise SystemExit(main())
