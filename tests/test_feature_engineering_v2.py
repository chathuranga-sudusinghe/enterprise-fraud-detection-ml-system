import warnings

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


def test_build_output_refactor_preserves_legacy_columns_values_and_frequency_order():
    train = make_training_frame()
    new_data = make_training_frame().copy()
    new_data["card1"] = [9999, 1111, 2222]
    new_data["P_emaildomain"] = ["new.example", "gmail.com", None]

    legacy_transformer = LegacyBuildOutputFeatureEngineeringV2()
    optimized_transformer = FeatureEngineeringV2()

    legacy_train = legacy_transformer.fit_transform(train)
    optimized_train = optimized_transformer.fit_transform(train)
    legacy_new = legacy_transformer.transform(new_data)
    optimized_new = optimized_transformer.transform(new_data)

    assert legacy_transformer.feature_names_ == optimized_transformer.feature_names_
    assert list(legacy_train.columns) == list(optimized_train.columns)
    assert list(legacy_new.columns) == list(optimized_new.columns)
    pd.testing.assert_frame_equal(legacy_train, optimized_train)
    pd.testing.assert_frame_equal(legacy_new, optimized_new)
    assert frequency_column_positions(legacy_train) == frequency_column_positions(
        optimized_train
    )


def test_wide_frequency_output_does_not_emit_performance_warning():
    transformer = FeatureEngineeringV2()
    wide_frequency_columns = [f"freq_extra_{idx}" for idx in range(140)]
    transformer.frequency_columns = [
        *transformer.frequency_columns,
        *wide_frequency_columns,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        output = transformer.fit_transform(make_wide_frequency_frame(wide_frequency_columns))

    assert output.columns.tolist() == transformer.feature_names_
    assert "freq_extra_0_frequency" in output.columns
    assert "freq_extra_139_frequency" in output.columns


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


def make_wide_frequency_frame(extra_columns: list[str]) -> pd.DataFrame:
    frame = make_training_frame()
    extra_frame = pd.DataFrame(
        {
            col: [f"{col}_a", f"{col}_b", f"{col}_{idx}"]
            for idx, col in enumerate(extra_columns)
        },
        index=frame.index,
    )
    return pd.concat([frame, extra_frame], axis=1)


def frequency_column_positions(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return [
        (column, idx)
        for idx, column in enumerate(frame.columns)
        if column.endswith("_frequency")
    ]


class LegacyBuildOutputFeatureEngineeringV2(FeatureEngineeringV2):
    def _build_output(self, X: pd.DataFrame) -> pd.DataFrame:
        model_columns = [
            col
            for col in X.columns
            if col not in self.id_columns
            and col not in self.ignored_columns
            and col != "uid_time_to_next"
        ]
        output = X[model_columns].copy()

        if self.frequency_enabled:
            for col, freq_map in self.frequency_maps_.items():
                if col in X.columns:
                    output[f"{col}_frequency"] = (
                        X[col].map(freq_map).fillna(self.frequency_unknown_value)
                    )

        self._ensure_no_duplicate_columns(output)
        return output
