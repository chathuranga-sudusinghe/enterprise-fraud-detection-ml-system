import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.inference.predict_v2 import EXPECTED_FEATURE_COUNT, FraudPredictorV2


def test_fraud_predictor_v2_loads_fake_artifacts_and_predicts(tmp_path):
    feature_columns = make_feature_columns()
    write_fake_v2_artifacts(tmp_path, feature_columns=feature_columns)

    predictor = FraudPredictorV2(artifact_dir=tmp_path)
    result = predictor.predict(pd.DataFrame([{"TransactionAmt": 100.0}]))

    assert result["fraud_probability"].iloc[0] == 0.91
    assert result["fraud_prediction"].iloc[0] == 1
    assert result["threshold"].iloc[0] == 0.10
    assert result["model_version"].iloc[0] == "v2"
    assert result["model_family"].iloc[0] == "catboost"
    assert result["feature_count"].iloc[0] == EXPECTED_FEATURE_COUNT


def test_fraud_predictor_v2_fails_closed_when_features_are_missing(tmp_path):
    feature_columns = make_feature_columns()
    write_fake_v2_artifacts(
        tmp_path,
        feature_columns=feature_columns,
        transformer=FakeMissingFeatureTransformer(feature_columns),
    )

    predictor = FraudPredictorV2(artifact_dir=tmp_path)

    with pytest.raises(ValueError, match="missing required columns"):
        predictor.predict(pd.DataFrame([{"TransactionAmt": 100.0}]))


def test_fraud_predictor_v2_rejects_wrong_metadata(tmp_path):
    feature_columns = make_feature_columns()
    metadata = make_metadata()
    metadata["model_family"] = "lightgbm"
    write_fake_v2_artifacts(
        tmp_path,
        feature_columns=feature_columns,
        metadata=metadata,
    )

    predictor = FraudPredictorV2(artifact_dir=tmp_path)

    with pytest.raises(ValueError, match="model_family"):
        predictor.predict(pd.DataFrame([{"TransactionAmt": 100.0}]))


def write_fake_v2_artifacts(
    artifact_dir: Path,
    *,
    feature_columns: list[str],
    metadata: dict[str, object] | None = None,
    transformer=None,
) -> None:
    transformer = transformer or FakeTransformer(feature_columns)
    metadata = metadata or make_metadata()

    joblib.dump(FakeModel(), artifact_dir / "fraud_catboost_v2.joblib")
    joblib.dump(transformer, artifact_dir / "feature_transformer_v2.joblib")
    (artifact_dir / "feature_columns_v2.json").write_text(
        json.dumps(feature_columns),
        encoding="utf-8",
    )
    (artifact_dir / "metadata_v2.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (artifact_dir / "threshold_v2.json").write_text(
        json.dumps({"threshold": 0.10}),
        encoding="utf-8",
    )


def make_feature_columns() -> list[str]:
    return [f"feature_{idx}" for idx in range(EXPECTED_FEATURE_COUNT)]


def make_metadata() -> dict[str, object]:
    return {
        "model_version": "v2",
        "model_family": "catboost",
        "n_features": EXPECTED_FEATURE_COUNT,
        "threshold": 0.10,
    }


class FakeTransformer:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def transform(self, input_df):
        return pd.DataFrame(
            {
                feature: [float(idx)]
                for idx, feature in enumerate(reversed(self.feature_columns))
            }
        )


class FakeMissingFeatureTransformer:
    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def transform(self, input_df):
        return pd.DataFrame(
            {
                feature: [float(idx)]
                for idx, feature in enumerate(self.feature_columns[1:])
            }
        )


class FakeModel:
    def predict_proba(self, input_df):
        return np.array([[0.09, 0.91]])
