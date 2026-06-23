import pandas as pd
import pytest

from ml.training.feature_engineering_v2 import FeatureEngineeringV2


def make_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [3600, 7200, 90000],
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


def test_fit_transform_returns_dataframe():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert isinstance(output, pd.DataFrame)


def test_missing_required_raw_columns_raise_value_error():
    transformer = FeatureEngineeringV2()
    df = make_training_frame().drop(columns=["TransactionDT"])

    with pytest.raises(ValueError, match="Missing required raw columns"):
        transformer.fit(df)


def test_numeric_missing_values_are_filled_with_train_medians():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output.loc[1, "TransactionAmt"] == 200.0


def test_numeric_missing_indicator_is_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "TransactionAmt_was_missing" in output.columns
    assert output["TransactionAmt_was_missing"].tolist() == [0, 1, 0]


def test_categorical_missing_values_become_missing_token():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output.loc[1, "card4"] == "__MISSING__"
    assert output.loc[2, "P_emaildomain"] == "__MISSING__"


def test_unseen_categories_during_transform_become_unknown_token():
    transformer = FeatureEngineeringV2()
    transformer.fit(make_training_frame())
    new_data = make_training_frame().iloc[[0]].copy()
    new_data["card4"] = "amex"
    new_data["P_emaildomain"] = "new.example"

    output = transformer.transform(new_data)

    assert output["card4"].iloc[0] == "__UNKNOWN__"
    assert output["P_emaildomain"].iloc[0] == "__UNKNOWN__"


def test_safe_time_features_are_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "transaction_hour" in output.columns
    assert "transaction_day" in output.columns
    assert output["transaction_hour"].tolist() == [1.0, 2.0, 1.0]
    assert output["transaction_day"].tolist() == [0.0, 0.0, 1.0]


def test_uid_time_to_next_is_not_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "uid_time_to_next" not in output.columns


def test_output_columns_are_deterministic_and_ordered():
    transformer = FeatureEngineeringV2()
    train = make_training_frame()

    first = transformer.fit_transform(train)
    second = transformer.transform(train)

    assert list(first.columns) == transformer.feature_names_
    assert list(second.columns) == transformer.feature_names_


def test_no_duplicate_output_columns():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert len(output.columns) == len(set(output.columns))


def test_target_column_is_not_included_in_output():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "isFraud" not in output.columns


def test_transaction_id_does_not_leak_into_output():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "TransactionID" not in output.columns


def test_transform_uses_fit_time_medians_category_maps_and_frequency_maps():
    transformer = FeatureEngineeringV2()
    transformer.fit(make_training_frame())
    new_data = pd.DataFrame(
        {
            "TransactionID": [4],
            "TransactionDT": [180000],
            "TransactionAmt": [None],
            "card1": [9999],
            "card2": [None],
            "card3": [150.0],
            "card4": ["discover"],
            "addr1": [999.0],
            "P_emaildomain": ["new.example"],
            "isFraud": [1],
        }
    )

    output = transformer.transform(new_data)

    assert output["TransactionAmt"].iloc[0] == 200.0
    assert output["card2"].iloc[0] == 15.0
    assert output["card4"].iloc[0] == "__UNKNOWN__"
    assert output["P_emaildomain"].iloc[0] == "__UNKNOWN__"
    assert output["card1_frequency"].iloc[0] == 0
    assert output["addr1_frequency"].iloc[0] == 0
