import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from ml.inference.predict import FraudPredictor
from ml.training.feature_engineering import FraudFeatureEngineeringEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "model_artifacts"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns_v1.json"
TRANSFORMER_PATH = ARTIFACT_DIR / "feature_transformer_v1.joblib"
MODEL_PATH = ARTIFACT_DIR / "fraud_lgbm_v1.joblib"

EXPECTED_CONTRACT_FEATURE_COUNT = 445
EXPECTED_FRESH_FEATURE_COUNT = 438
MISSING_FROM_FRESH_ENGINEERING = {
    "uid_time_to_next",
    "uid_time_from_prev",
    "uid_txn_count",
    "uid_amt_mean",
    "uid_amt_std",
    "uid_amt_median",
    "uid_amt_deviation",
}
ENGINEERED_FEATURES = {
    "day",
    "hour",
    "card1_freq",
    "card2_freq",
    "card3_freq",
    "card4_freq",
    *MISSING_FROM_FRESH_ENGINEERING,
}
CATEGORICAL_CONTRACT_FEATURES = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
}


@pytest.fixture(scope="session")
def persisted_feature_columns():
    with FEATURE_COLUMNS_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def persisted_transformer():
    return joblib.load(TRANSFORMER_PATH)


@pytest.fixture(scope="session")
def persisted_model():
    return joblib.load(MODEL_PATH)


@pytest.fixture()
def representative_raw_input(persisted_feature_columns):
    raw_columns = [
        col for col in persisted_feature_columns if col not in ENGINEERED_FEATURES
    ]
    data = {}

    for idx, col in enumerate(raw_columns):
        if col in CATEGORICAL_CONTRACT_FEATURES:
            data[col] = [f"{col}_a", f"{col}_b"]
        else:
            data[col] = [float(idx + 1), float(idx + 2)]

    data.update(
        {
            "TransactionDT": [86_400, 172_800],
            "TransactionAmt": [100.0, 250.0],
            "card1": [12_345, 12_345],
            "card2": [111.0, 111.0],
            "card3": [150.0, 150.0],
            "card4": ["visa", "visa"],
            "addr1": [100.0, 100.0],
        }
    )

    return pd.DataFrame(data, columns=raw_columns)


@pytest.fixture()
def fresh_feature_output(representative_raw_input):
    engine = FraudFeatureEngineeringEngine()
    engine.fit(representative_raw_input)
    return engine.transform(representative_raw_input)


def test_persisted_feature_json_is_445_ordered_unique_names(
    persisted_feature_columns,
):
    assert len(persisted_feature_columns) == EXPECTED_CONTRACT_FEATURE_COUNT, (
        "Persisted feature JSON must remain the 445-feature runtime contract."
    )
    assert len(set(persisted_feature_columns)) == len(persisted_feature_columns), (
        "Persisted feature JSON must not contain duplicate feature names."
    )
    assert persisted_feature_columns[-7:] == [
        "uid_time_to_next",
        "uid_time_from_prev",
        "uid_txn_count",
        "uid_amt_mean",
        "uid_amt_std",
        "uid_amt_median",
        "uid_amt_deviation",
    ], "Persisted feature order changed at the contract-sensitive UID tail."


def test_persisted_transformer_schema_matches_feature_json(
    persisted_feature_columns,
    persisted_transformer,
):
    assert persisted_transformer.feature_schema == persisted_feature_columns, (
        "Serialized transformer feature_schema must match persisted feature JSON "
        "exactly, including order."
    )


def test_lightgbm_model_feature_order_matches_persisted_contract(
    persisted_feature_columns,
    persisted_model,
):
    booster_feature_names = persisted_model.booster_.feature_name()

    assert persisted_model.n_features_in_ == EXPECTED_CONTRACT_FEATURE_COUNT, (
        "Persisted LightGBM model must continue to expect 445 features."
    )
    assert booster_feature_names == persisted_feature_columns, (
        "Persisted LightGBM booster feature names must match the persisted "
        "feature contract exactly, including order."
    )


def test_fresh_feature_engineering_documents_current_438_feature_contract(
    persisted_feature_columns,
    fresh_feature_output,
):
    fresh_columns = list(fresh_feature_output.columns)
    missing_from_fresh = set(persisted_feature_columns) - set(fresh_columns)
    unexpected_fresh = set(fresh_columns) - set(persisted_feature_columns)

    assert len(fresh_columns) == EXPECTED_FRESH_FEATURE_COUNT, (
        "Fresh current-source feature engineering is expected to produce 438 "
        "features until the persisted contract is intentionally reconciled."
    )
    assert missing_from_fresh == MISSING_FROM_FRESH_ENGINEERING, (
        "Fresh feature engineering missing-feature set changed; this may affect "
        "the documented 445-vs-438 contract gap."
    )
    assert unexpected_fresh == set(), (
        "Fresh feature engineering produced features outside the persisted "
        f"contract: {sorted(unexpected_fresh)}"
    )
    assert fresh_columns == [
        col for col in persisted_feature_columns if col not in missing_from_fresh
    ], "Fresh feature order changed relative to the persisted contract."


def test_inference_alignment_returns_445_ordered_features_without_writes(
    persisted_feature_columns,
    representative_raw_input,
):
    predictor = FraudPredictor()

    transformed = predictor.feature_engine.transform(representative_raw_input)
    aligned = transformed[predictor.feature_columns]

    assert predictor.feature_columns == persisted_feature_columns, (
        "Inference feature_columns must load the persisted feature JSON order."
    )
    assert list(aligned.columns) == persisted_feature_columns, (
        "Inference alignment must return the exact 445-feature persisted order."
    )
    assert aligned.shape[1] == EXPECTED_CONTRACT_FEATURE_COUNT, (
        "Inference alignment must keep the model input at 445 features."
    )


def test_time_direction_features_use_existing_zero_fill_fallback(
    persisted_transformer,
    representative_raw_input,
):
    transformed = persisted_transformer.transform(representative_raw_input)

    for feature in ["uid_time_to_next", "uid_time_from_prev"]:
        assert feature in transformed.columns, (
            f"{feature} must remain present in the persisted inference contract."
        )
        assert transformed[feature].eq(0).all(), (
            f"{feature} is currently expected to use the existing zero-fill "
            "fallback rather than being recomputed."
        )
