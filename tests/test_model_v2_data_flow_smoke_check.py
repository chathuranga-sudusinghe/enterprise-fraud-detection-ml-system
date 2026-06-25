import pandas as pd

import scripts.smoke_check_model_v2_data_flow as smoke_check


def make_merged_frame() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append(
            {
                "TransactionID": i + 1,
                "TransactionDT": (20 - i) * 100,
                "TransactionAmt": float(100 + i),
                "card1": 1000 + (i % 3),
                "card2": 200 + (i % 2),
                "card3": 150,
                "card4": "visa" if i % 2 == 0 else "mastercard",
                "addr1": 10 + (i % 4),
                "DeviceType": "desktop" if i % 2 == 0 else None,
                "isFraud": i % 2,
            }
        )
    return pd.DataFrame(rows)


def test_smoke_check_runs_merge_split_and_dry_run(monkeypatch):
    calls = []
    merged = make_merged_frame()

    def fake_load_transaction_identity_dataset():
        calls.append("merge")
        return merged

    def fake_prepare_time_based_train_val_test_split(df):
        calls.append("split")
        assert df is merged
        sorted_df = df.sort_values("TransactionDT", kind="mergesort").reset_index(
            drop=True
        )
        train = sorted_df.iloc[:14].drop(columns=["isFraud"])
        val = sorted_df.iloc[14:17].drop(columns=["isFraud"])
        test = sorted_df.iloc[17:].drop(columns=["isFraud"])
        return {
            "X_train": train,
            "y_train": sorted_df.iloc[:14]["isFraud"],
            "X_val": val,
            "y_val": sorted_df.iloc[14:17]["isFraud"],
            "X_test": test,
            "y_test": sorted_df.iloc[17:]["isFraud"],
        }

    def fake_run_training_pipeline_v2_dry_run(**kwargs):
        calls.append("dry_run")
        assert len(kwargs["X_train"]) == 14
        assert len(kwargs["X_val"]) == 3
        assert len(kwargs["X_test"]) == 3
        return {
            "feature_count": 42,
            "would_write_artifacts": False,
            "transformer_class": "FeatureEngineeringV2",
        }

    monkeypatch.setattr(
        smoke_check,
        "load_transaction_identity_dataset",
        fake_load_transaction_identity_dataset,
    )
    monkeypatch.setattr(
        smoke_check,
        "prepare_time_based_train_val_test_split",
        fake_prepare_time_based_train_val_test_split,
    )
    monkeypatch.setattr(
        smoke_check,
        "run_training_pipeline_v2_dry_run",
        fake_run_training_pipeline_v2_dry_run,
    )

    summary = smoke_check.run_model_v2_data_flow_smoke_check()

    assert calls == ["merge", "split", "dry_run"]
    assert summary["merged_shape"] == (20, 10)
    assert summary["split_row_counts"] == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }
    assert summary["feature_count"] == 42
    assert summary["would_write_artifacts"] is False
    assert summary["transformer_class"] == "FeatureEngineeringV2"
