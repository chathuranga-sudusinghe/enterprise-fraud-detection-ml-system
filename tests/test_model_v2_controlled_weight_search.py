import math

import numpy as np
import pandas as pd
import pytest

import ml.experiments.model_v2_controlled_weight_search as weight_search
from ml.experiments.model_v2_controlled_weight_search import (
    generate_controlled_scale_pos_weight_candidates,
    run_model_v2_controlled_weight_search_experiment,
)


def test_generate_controlled_scale_pos_weight_candidates_includes_sqrt_full_weight():
    candidates = generate_controlled_scale_pos_weight_candidates(27.4343)

    assert candidates == [1.0, 3.0, 5.0, round(math.sqrt(27.4343), 4), 10.0]


def test_generate_controlled_scale_pos_weight_candidates_removes_duplicates():
    candidates = generate_controlled_scale_pos_weight_candidates(9.0)

    assert candidates == [1.0, 3.0, 5.0, 10.0]


def test_generate_controlled_scale_pos_weight_candidates_rejects_invalid_full_weight():
    with pytest.raises(ValueError, match="positive"):
        generate_controlled_scale_pos_weight_candidates(0)


def test_controlled_weight_search_trains_one_candidate_per_weight(monkeypatch):
    train_calls = []

    monkeypatch.setattr(
        weight_search,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: pd.DataFrame({"unused": []}),
    )
    monkeypatch.setattr(
        weight_search,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(weight_search, "FeatureEngineeringV2", FakeTransformer)

    def fake_train_lightgbm_v2(
        X_train,
        y_train,
        X_val,
        y_val,
        categorical_cols,
        scale_pos_weight=None,
    ):
        train_calls.append(scale_pos_weight)
        offset = 0.0 if scale_pos_weight is None else 0.05
        return FakeModel([0.05 + offset, 0.80 + offset]), np.array(
            [0.10 + offset, 0.70 + offset]
        )

    monkeypatch.setattr(weight_search, "train_lightgbm_v2", fake_train_lightgbm_v2)

    summary = run_model_v2_controlled_weight_search_experiment(
        candidate_weights=[1.0, 3.0],
    )

    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["full_scale_pos_weight"] == 3.0
    assert summary["candidate_weights"] == [1.0, 3.0]
    assert train_calls == [None, 3.0]
    assert len(summary["candidates"]) == 2
    assert summary["candidates"][0]["trainer_scale_pos_weight"] is None
    assert summary["candidates"][1]["trainer_scale_pos_weight"] == 3.0
    assert summary["candidate_summary"][0]["scale_pos_weight"] == 1.0
    assert summary["candidates"][0]["validation_false_negative_analysis"]


class FakeTransformer:
    categorical_columns_ = []
    feature_names_ = ["feature"]

    def fit_transform(self, X):
        return X[["feature"]].copy()

    def transform(self, X):
        return X[["feature"]].copy()


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        probabilities = self.probabilities[: len(X)]
        return np.array([[1 - probability, probability] for probability in probabilities])


def _synthetic_splits():
    return {
        "X_train": pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}),
        "y_train": pd.Series([0, 0, 0, 1]),
        "X_val": pd.DataFrame({"feature": [5.0, 6.0]}),
        "y_val": pd.Series([0, 1]),
        "X_test": pd.DataFrame({"feature": [7.0, 8.0]}),
        "y_test": pd.Series([0, 1]),
    }
