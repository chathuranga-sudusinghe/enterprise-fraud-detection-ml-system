from pathlib import Path

import pandas as pd
import pytest

import ml.pipelines.training_pipeline_v2 as training_pipeline_v2
from ml.pipelines.training_pipeline_v2 import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TRANSACTION_PATH,
    KNOWN_V1_ARTIFACT_NAMES,
    V2_ARTIFACT_PATHS,
    load_full_dataset,
    load_transaction_identity_dataset,
    merge_transaction_identity,
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


def make_transaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [400, 100, 300, 200],
            "TransactionAmt": [100.0, 200.0, 300.0, 400.0],
            "card1": [1001, 1002, 1003, 1004],
            "card2": [201, 202, 203, 204],
            "card3": [150, 150, 150, 150],
            "card4": ["visa", "mastercard", "visa", "visa"],
            "addr1": [10, 20, 30, 40],
            "isFraud": [0, 1, 0, 1],
        }
    )


def make_identity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 3, 999],
            "DeviceType": ["desktop", "mobile", "desktop"],
            "DeviceInfo": ["windows", "ios", "unused"],
        }
    )


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


def test_transaction_identity_left_merge_keeps_all_transaction_rows():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())

    assert len(merged) == len(make_transaction_frame())
    assert list(merged["TransactionID"]) == [1, 2, 3, 4]


def test_unmatched_identity_rows_do_not_drop_transaction_rows():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())

    assert 999 not in set(merged["TransactionID"])
    assert set(merged["TransactionID"]) == {1, 2, 3, 4}


def test_identity_columns_are_added_for_matching_ids():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())

    matched = merged.set_index("TransactionID")
    assert matched.loc[1, "DeviceType"] == "desktop"
    assert matched.loc[3, "DeviceInfo"] == "ios"


def test_missing_identity_values_remain_nan_for_non_matching_transaction_ids():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())
    unmatched = merged.set_index("TransactionID")

    assert pd.isna(unmatched.loc[2, "DeviceType"])
    assert pd.isna(unmatched.loc[4, "DeviceInfo"])


def test_merge_preserves_target_and_transaction_time_columns():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())

    assert "isFraud" in merged.columns
    assert "TransactionDT" in merged.columns


def test_missing_transaction_id_in_transaction_raises_value_error():
    transaction_df = make_transaction_frame().drop(columns=["TransactionID"])

    with pytest.raises(ValueError, match="transaction dataset"):
        merge_transaction_identity(transaction_df, make_identity_frame())


def test_missing_transaction_id_in_identity_raises_value_error():
    identity_df = make_identity_frame().drop(columns=["TransactionID"])

    with pytest.raises(ValueError, match="identity dataset"):
        merge_transaction_identity(make_transaction_frame(), identity_df)


def test_missing_target_in_transaction_raises_value_error():
    transaction_df = make_transaction_frame().drop(columns=["isFraud"])

    with pytest.raises(ValueError, match="target column"):
        merge_transaction_identity(transaction_df, make_identity_frame())


def test_target_present_in_identity_raises_value_error():
    identity_df = make_identity_frame()
    identity_df["isFraud"] = [0, 0, 1]

    with pytest.raises(ValueError, match="must not contain target column"):
        merge_transaction_identity(make_transaction_frame(), identity_df)


def test_merged_dataset_is_compatible_with_time_based_split():
    merged = merge_transaction_identity(make_transaction_frame(), make_identity_frame())

    splits = prepare_time_based_train_val_test_split(
        merged,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
    )

    assert len(splits["X_train"]) == 2
    assert len(splits["X_val"]) == 1
    assert len(splits["X_test"]) == 1
    assert "isFraud" not in splits["X_train"].columns
    assert "DeviceType" in splits["X_train"].columns


def test_default_transaction_path_points_to_raw_transaction_parquet():
    assert DEFAULT_TRANSACTION_PATH.as_posix().endswith(
        "lakehouse/raw/train_transaction.parquet"
    )


def test_default_identity_path_points_to_raw_identity_parquet():
    assert DEFAULT_IDENTITY_PATH.as_posix().endswith(
        "lakehouse/raw/train_identity.parquet"
    )


def test_load_transaction_identity_dataset_uses_default_paths(monkeypatch):
    loaded_paths = []

    def fake_load_full_dataset(path):
        loaded_paths.append(Path(path))
        if Path(path) == DEFAULT_TRANSACTION_PATH:
            return make_transaction_frame()
        if Path(path) == DEFAULT_IDENTITY_PATH:
            return make_identity_frame()
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(
        training_pipeline_v2,
        "load_full_dataset",
        fake_load_full_dataset,
    )

    merged = load_transaction_identity_dataset()

    assert loaded_paths == [DEFAULT_TRANSACTION_PATH, DEFAULT_IDENTITY_PATH]
    assert len(merged) == len(make_transaction_frame())
    assert "DeviceType" in merged.columns


def test_load_full_dataset_supports_custom_csv_path(tmp_path):
    csv_path = tmp_path / "custom.csv"
    expected = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
    expected.to_csv(csv_path, index=False)

    loaded = load_full_dataset(csv_path)

    assert loaded.equals(expected)


def test_load_full_dataset_supports_custom_parquet_path(tmp_path):
    parquet_path = tmp_path / "custom.parquet"
    expected = pd.DataFrame({"TransactionID": [1], "isFraud": [0]})
    expected.to_parquet(parquet_path, index=False)

    loaded = load_full_dataset(parquet_path)

    assert loaded.equals(expected)
