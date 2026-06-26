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
            "card5": [226.0, 226.0, None],
            "card6": ["debit", "credit", None],
            "addr1": [100.0, 100.0, 200.0],
            "addr2": [87.0, 87.0, None],
            "id_12": [1, 1, 2],
            "id_13": [None, 10, None],
            "P_emaildomain": ["gmail.com", "yahoo.com", None],
            "R_emaildomain": [None, "gmail.com", None],
            "ProductCD": ["W", "C", "W"],
            "DeviceType": [None, "mobile", None],
            "DeviceInfo": [None, "ios", "SM-G960U"],
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
            "TransactionAmt": ["not-a-number"],
            "card1": [9999],
            "card2": [None],
            "card3": [150.0],
            "card4": ["discover"],
            "card5": [999.0],
            "card6": ["charge"],
            "addr1": [999.0],
            "addr2": [999.0],
            "id_12": [999],
            "id_13": [None],
            "P_emaildomain": ["new.example"],
            "R_emaildomain": ["new.example"],
            "ProductCD": ["S"],
            "DeviceType": [None],
            "DeviceInfo": ["new-device"],
            "isFraud": [1],
        }
    )

    output = transformer.transform(new_data)

    assert output["TransactionAmt"].iloc[0] == 200.0
    assert output["TransactionAmt_was_missing"].iloc[0] == 1
    assert output["card2"].iloc[0] == "__MISSING__"
    assert output["card4"].iloc[0] == "__UNKNOWN__"
    assert output["id_12"].iloc[0] == "__UNKNOWN__"
    assert output["P_emaildomain"].iloc[0] == "__UNKNOWN__"
    assert output["card1_frequency"].iloc[0] == 0
    assert output["addr1_frequency"].iloc[0] == 0


def test_identifier_like_numeric_columns_are_categorical():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert {"card1", "card2", "card3", "addr1", "id_12"}.issubset(
        set(transformer.categorical_columns_)
    )
    assert output["card1"].dtype == object
    assert output["addr1"].dtype == object
    assert output["id_12"].dtype == object


def test_true_numeric_columns_remain_numeric():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert "TransactionAmt" in transformer.numerical_columns_
    assert "TransactionDT" in transformer.numerical_columns_
    assert pd.api.types.is_numeric_dtype(output["TransactionAmt"])


def test_train_val_test_output_columns_remain_consistent():
    transformer = FeatureEngineeringV2()
    train = make_training_frame()
    val = make_training_frame().copy()
    test = make_training_frame().copy()
    val["card1"] = [9999, 1111, 2222]
    test["P_emaildomain"] = ["new.example", "gmail.com", None]

    train_out = transformer.fit_transform(train)
    val_out = transformer.transform(val)
    test_out = transformer.transform(test)

    assert list(train_out.columns) == list(val_out.columns) == list(test_out.columns)


def test_false_negative_driven_missing_flags_are_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output["DeviceInfo_missing_flag"].tolist() == [1, 0, 0]
    assert output["DeviceType_missing_flag"].tolist() == [1, 0, 1]
    assert output["R_emaildomain_missing_flag"].tolist() == [1, 0, 1]
    assert output["P_emaildomain_missing_flag"].tolist() == [0, 0, 1]


def test_identity_missingness_features_are_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output["identity_missing_count"].tolist() == [1, 0, 1]
    assert output["identity_missing_ratio"].tolist() == [0.5, 0.0, 0.5]
    assert output["high_identity_missing_flag"].tolist() == [0, 0, 0]


def test_product_and_missingness_interaction_features_are_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output["ProductCD_W_flag"].tolist() == [1, 0, 1]
    assert output["ProductCD_DeviceType_missing_interaction"].tolist() == [
        "W__DeviceType_missing_1",
        "C__DeviceType_missing_0",
        "W__DeviceType_missing_1",
    ]
    assert output["ProductCD_DeviceInfo_missing_interaction"].tolist() == [
        "W__DeviceInfo_missing_1",
        "C__DeviceInfo_missing_0",
        "W__DeviceInfo_missing_0",
    ]
    assert output["ProductCD_R_emaildomain_missing_interaction"].tolist() == [
        "W__R_emaildomain_missing_1",
        "C__R_emaildomain_missing_0",
        "W__R_emaildomain_missing_1",
    ]


def test_card_interaction_features_are_created():
    transformer = FeatureEngineeringV2()

    output = transformer.fit_transform(make_training_frame())

    assert output["card3_addr2_interaction"].tolist() == [
        "150.0__87.0",
        "150.0__87.0",
        "150.0____MISSING__",
    ]
    assert output["card4_card6_interaction"].tolist() == [
        "visa__debit",
        "__MISSING____credit",
        "mastercard____MISSING__",
    ]


def test_false_negative_driven_interactions_are_categorical_and_unknown_mapped():
    transformer = FeatureEngineeringV2()
    transformer.fit(make_training_frame())
    new_data = make_training_frame().iloc[[0]].copy()
    new_data["ProductCD"] = "S"
    new_data["DeviceType"] = None
    new_data["DeviceInfo"] = None
    new_data["R_emaildomain"] = None
    new_data["card3"] = 999.0
    new_data["addr2"] = 999.0
    new_data["card4"] = "amex"
    new_data["card6"] = "charge"

    output = transformer.transform(new_data)

    assert "card3_addr2_interaction" in transformer.categorical_columns_
    assert output["ProductCD_DeviceType_missing_interaction"].iloc[0] == "__UNKNOWN__"
    assert output["card3_addr2_interaction"].iloc[0] == "__UNKNOWN__"
    assert output["card4_card6_interaction"].iloc[0] == "__UNKNOWN__"
