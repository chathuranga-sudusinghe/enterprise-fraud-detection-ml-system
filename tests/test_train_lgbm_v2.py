import numpy as np
import pandas as pd
import pytest

import ml.training.train_lgbm_v2 as train_lgbm_v2_module
from ml.training.train_lgbm_v2 import train_lightgbm_v2


def test_train_lightgbm_v2_default_does_not_set_scale_pos_weight(monkeypatch):
    created_params = []

    monkeypatch.setattr(
        train_lgbm_v2_module,
        "LGBMClassifier",
        _fake_lgbm_classifier(created_params),
    )

    train_lightgbm_v2(
        X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
        y_train=pd.Series([0, 1]),
        X_val=pd.DataFrame({"feature": [3.0, 4.0]}),
        y_val=pd.Series([0, 1]),
        categorical_cols=[],
    )

    assert "scale_pos_weight" not in created_params[0]


def test_train_lightgbm_v2_sets_scale_pos_weight_when_explicit(monkeypatch):
    created_params = []

    monkeypatch.setattr(
        train_lgbm_v2_module,
        "LGBMClassifier",
        _fake_lgbm_classifier(created_params),
    )

    train_lightgbm_v2(
        X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
        y_train=pd.Series([0, 1]),
        X_val=pd.DataFrame({"feature": [3.0, 4.0]}),
        y_val=pd.Series([0, 1]),
        categorical_cols=[],
        scale_pos_weight=3.5,
    )

    assert created_params[0]["scale_pos_weight"] == 3.5


def test_train_lightgbm_v2_allows_empty_categorical_columns(monkeypatch):
    created_params = []

    monkeypatch.setattr(
        train_lgbm_v2_module,
        "LGBMClassifier",
        _fake_lgbm_classifier(created_params),
    )

    model, probabilities = train_lightgbm_v2(
        X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
        y_train=pd.Series([0, 1]),
        X_val=pd.DataFrame({"feature": [3.0, 4.0]}),
        y_val=pd.Series([0, 1]),
        categorical_cols=[],
    )

    assert model.fit_called is True
    assert probabilities.tolist() == [0.2, 0.8]


def test_train_lightgbm_v2_rejects_missing_categorical_columns(monkeypatch):
    created_params = []

    monkeypatch.setattr(
        train_lgbm_v2_module,
        "LGBMClassifier",
        _fake_lgbm_classifier(created_params),
    )

    with pytest.raises(ValueError, match="Categorical columns missing"):
        train_lightgbm_v2(
            X_train=pd.DataFrame({"feature": [1.0, 2.0]}),
            y_train=pd.Series([0, 1]),
            X_val=pd.DataFrame({"feature": [3.0, 4.0]}),
            y_val=pd.Series([0, 1]),
            categorical_cols=["missing_category"],
        )


def _fake_lgbm_classifier(created_params):
    class FakeLGBMClassifier:
        def __init__(self, **params):
            created_params.append(params)
            self.best_iteration_ = 1
            self.best_score_ = {"valid_0": {"auc": 0.5}}
            self.fit_called = False

        def fit(self, *args, **kwargs):
            self.fit_called = True
            return self

        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.2, 0.8]])[: len(X)]

    return FakeLGBMClassifier
