from pathlib import Path

import pandas as pd
import pytest

from ml.pipelines.training_pipeline_v2 import (
    KNOWN_V1_ARTIFACT_NAMES,
    V2_ARTIFACT_PATHS,
    prepare_time_based_train_val_test_split,
    run_training_pipeline_v2_dry_run,
    split_by_time_order,
    validate_v2_artifact_paths,
)


def make_frame(offset: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1 + offset, 2 + offset, 3 + offset],
            "TransactionDT": [3600 + offset, 7200 + offset, 90000 + offset],
            "TransactionAmt": [100.0, None, 300.0],
            "card1": [1111, 1111, 2222],
            "card2": [10.0, 20.0, None],
            "card3": [150.0, 150.0, 150.0],
            "card4": ["visa", None, "mastercard"],
            "addr1": [100.0, 100.0, 200.0],
            "P_emaildomain": ["gmail.com", "yahoo.com", None],
            "isFraud": [0, 1, 0],
        }
    )


def make_target() -> pd.Series:
    return pd.Series([0, 1, 0])


def run_dry_run() -> dict:
    return run_training_pipeline_v2_dry_run(
        X_train=make_frame(),
        y_train=make_target(),
        X_val=make_frame(offset=10),
        y_val=make_target(),
        X_test=make_frame(offset=20),
        y_test=make_target(),
    )


def make_unsorted_full_dataset(row_count: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(row_count):
        rows.append(
            {
                "TransactionID": i + 1,
                "TransactionDT": (row_count - i) * 100,
                "TransactionAmt": float(100 + i),
                "card1": 1000 + (i % 3),
                "card2": 200 + (i % 2),
                "card3": 150,
                "card4": "visa" if i % 2 == 0 else "mastercard",
                "addr1": 10 + (i % 4),
                "P_emaildomain": "gmail.com" if i % 2 == 0 else "yahoo.com",
                "isFraud": i % 2,
            }
        )

    return pd.DataFrame(rows)


def test_dry_run_uses_feature_engineering_v2():
    summary = run_dry_run()

    assert summary["feature_engineering_version"] == "v2"
    assert summary["transformer_class"] == "FeatureEngineeringV2"


def test_dry_run_returns_train_val_and_test_shapes():
    summary = run_dry_run()

    assert summary["train_shape"][0] == 3
    assert summary["val_shape"][0] == 3
    assert summary["test_shape"][0] == 3


def test_feature_count_equals_feature_names_length():
    summary = run_dry_run()

    assert summary["feature_count"] == len(summary["feature_names"])


def test_artifact_paths_are_v2_paths():
    summary = run_dry_run()

    assert summary["artifact_paths"]
    for path in summary["artifact_paths"].values():
        assert "_v2" in Path(path).stem


def test_no_v1_artifact_paths_are_used():
    summary = run_dry_run()

    artifact_names = {Path(path).name for path in summary["artifact_paths"].values()}
    assert artifact_names.isdisjoint(KNOWN_V1_ARTIFACT_NAMES)


def test_safety_guard_rejects_v1_artifact_names():
    unsafe_paths = {
        "model": Path("model_artifacts/fraud_lgbm_v1.joblib"),
        **{name: path for name, path in V2_ARTIFACT_PATHS.items() if name != "model"},
    }

    with pytest.raises(ValueError, match="must not write or target v1 artifacts"):
        validate_v2_artifact_paths(unsafe_paths)


def test_dry_run_does_not_write_artifact_files():
    summary = run_dry_run()

    assert summary["would_write_artifacts"] is False
    for path in summary["artifact_paths"].values():
        assert not Path(path).exists()


def test_target_and_transaction_id_do_not_leak_into_feature_names():
    summary = run_dry_run()

    assert "isFraud" not in summary["feature_names"]
    assert "TransactionID" not in summary["feature_names"]


def test_v1_predict_behavior_is_not_touched_by_module():
    summary = run_dry_run()

    assert "planned_v2_route" not in summary
    assert all("/predict" not in path for path in summary["artifact_paths"].values())


def test_time_based_split_preserves_time_order():
    train_df, val_df, test_df = split_by_time_order(make_unsorted_full_dataset())

    assert train_df["TransactionDT"].is_monotonic_increasing
    assert val_df["TransactionDT"].is_monotonic_increasing
    assert test_df["TransactionDT"].is_monotonic_increasing


def test_time_based_split_boundaries_do_not_overlap_future_rows():
    train_df, val_df, test_df = split_by_time_order(make_unsorted_full_dataset())

    assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min()


def test_prepare_split_separates_targets_from_features():
    df = make_unsorted_full_dataset()
    splits = prepare_time_based_train_val_test_split(df)
    expected_y_train = (
        df.sort_values("TransactionDT", kind="mergesort")["isFraud"]
        .iloc[:14]
        .reset_index(drop=True)
    )

    assert splits["y_train"].reset_index(drop=True).equals(expected_y_train)
    assert "isFraud" not in splits["X_train"].columns
    assert "isFraud" not in splits["X_val"].columns
    assert "isFraud" not in splits["X_test"].columns


def test_missing_transaction_dt_raises_value_error():
    df = make_unsorted_full_dataset().drop(columns=["TransactionDT"])

    with pytest.raises(ValueError, match="TransactionDT"):
        split_by_time_order(df)


def test_missing_target_raises_value_error():
    df = make_unsorted_full_dataset().drop(columns=["isFraud"])

    with pytest.raises(ValueError, match="isFraud"):
        split_by_time_order(df)


def test_split_sizes_are_deterministic_with_default_ratios():
    train_df, val_df, test_df = split_by_time_order(make_unsorted_full_dataset())

    assert len(train_df) == 14
    assert len(val_df) == 3
    assert len(test_df) == 3


def test_prepared_time_split_is_compatible_with_dry_run():
    splits = prepare_time_based_train_val_test_split(make_unsorted_full_dataset())

    summary = run_training_pipeline_v2_dry_run(
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
    )

    assert summary["train_shape"][0] == 14
    assert summary["val_shape"][0] == 3
    assert summary["test_shape"][0] == 3
