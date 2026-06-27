import builtins

import numpy as np
import pandas as pd
import pytest

import ml.experiments.model_v2_deep_learning_baseline as dl_baseline
from ml.experiments.model_v2_deep_learning_baseline import (
    build_tabular_mlp,
    get_torch_availability,
    prepare_mlp_feature_matrices,
    run_model_v2_deep_learning_baseline,
    select_best_neural_threshold,
)


def test_torch_availability_reports_missing_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    availability = get_torch_availability()

    assert availability["available"] is False
    assert "torch" in availability["reason"]
    assert availability["cuda_available"] is False


def test_build_tabular_mlp_constructs_binary_logit_model():
    torch = pytest.importorskip("torch")

    model = build_tabular_mlp(input_dim=7, hidden_dims=(5, 3), dropout=0.10)
    output = model(torch.zeros((2, 7), dtype=torch.float32))

    assert output.shape == (2, 1)


def test_select_best_neural_threshold_uses_alert_constraint_and_recall_priority():
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.95, 0.80, 0.20, 0.70, 0.15, 0.05])

    result = select_best_neural_threshold(
        y_true=y_true,
        y_proba=y_proba,
        max_alert_rate=0.50,
    )

    assert result["selected_threshold"] == 0.80
    assert result["selected_metrics"]["recall"] == pytest.approx(2 / 3)
    assert result["selected_metrics"]["alert_rate"] <= 0.50


def test_prepare_mlp_feature_matrices_uses_compact_category_codes(monkeypatch):
    def fail_get_dummies(*args, **kwargs):
        raise AssertionError("pd.get_dummies must not be used for MLP matrices")

    monkeypatch.setattr(pd, "get_dummies", fail_get_dummies)
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "category": ["known_a", "known_b", "known_a"],
            "high_cardinality": ["id_1", "id_2", "id_3"],
        }
    )
    val = pd.DataFrame(
        {
            "numeric": [4.0],
            "category": ["new_value"],
            "high_cardinality": ["id_999"],
        }
    )
    test = pd.DataFrame(
        {
            "numeric": [5.0],
            "category": ["known_b"],
            "high_cardinality": ["id_2"],
        }
    )

    matrices = prepare_mlp_feature_matrices(
        X_train=train,
        X_val=val,
        X_test=test,
    )

    assert matrices["input_dim"] == train.shape[1]
    assert matrices["X_train"].shape[1] == train.shape[1]
    assert matrices["X_val"].shape[1] == train.shape[1]
    assert matrices["X_test"].shape[1] == train.shape[1]
    assert matrices["X_train"].dtype == np.float32
    assert matrices["X_val"].dtype == np.float32
    assert matrices["X_test"].dtype == np.float32
    assert matrices["encoded_feature_names"] == list(train.columns)
    assert matrices["X_val"][0, 1] == 0.0
    assert matrices["X_val"][0, 2] == 0.0
    assert matrices["X_test"][0, 1] > 0.0


def test_deep_learning_baseline_skips_when_torch_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        dl_baseline,
        "get_torch_availability",
        lambda: {
            "available": False,
            "reason": "missing torch",
            "version": None,
            "cuda_available": False,
            "device": None,
        },
    )

    summary = run_model_v2_deep_learning_baseline()

    assert summary["status"] == "skipped"
    assert summary["artifacts_written"] is False
    assert summary["write_artifacts"] is False
    assert summary["decision"]["beats_catboost_baseline"] is False


def test_deep_learning_baseline_trained_path_is_non_artifact_workflow(monkeypatch):
    monkeypatch.setattr(
        dl_baseline,
        "get_torch_availability",
        lambda: {
            "available": True,
            "reason": "available",
            "version": "test",
            "cuda_available": False,
            "device": "cpu",
        },
    )
    monkeypatch.setattr(
        dl_baseline,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: None,
    )
    monkeypatch.setattr(
        dl_baseline,
        "prepare_time_based_train_val_test_split",
        lambda merged: _synthetic_splits(),
    )
    monkeypatch.setattr(dl_baseline, "FeatureEngineeringV2", FakeTransformer)
    monkeypatch.setattr(
        dl_baseline,
        "train_tabular_mlp",
        lambda **kwargs: {
            "model": object(),
            "device": "cpu",
            "val_proba": np.array([0.10, 0.90, 0.20, 0.80]),
            "epochs_completed": 2,
            "best_validation_pr_auc": 0.80,
        },
    )
    monkeypatch.setattr(
        dl_baseline,
        "predict_tabular_mlp",
        lambda **kwargs: np.array([0.05, 0.85, 0.30, 0.75]),
    )

    summary = run_model_v2_deep_learning_baseline()

    assert summary["status"] == "trained"
    assert summary["write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["feature_count"] == 2
    assert summary["mlp_input_dim"] == summary["feature_count"]
    assert summary["selected_threshold"] >= 0.10
    assert "/predict" not in str(summary)


class FakeTransformer:
    categorical_columns_ = ["category"]
    feature_names_ = ["feature", "category"]

    def fit_transform(self, X):
        return X[["feature", "category"]].copy()

    def transform(self, X):
        return X[["feature", "category"]].copy()


def _synthetic_splits():
    return {
        "X_train": pd.DataFrame(
            {
                "feature": [1.0, 2.0, 3.0, 4.0],
                "category": ["a", "b", "a", "c"],
            }
        ),
        "y_train": pd.Series([0, 1, 0, 1]),
        "X_val": pd.DataFrame(
            {
                "feature": [5.0, 6.0, 7.0, 8.0],
                "category": ["a", "b", "z", "c"],
            }
        ),
        "y_val": pd.Series([0, 1, 0, 1]),
        "X_test": pd.DataFrame(
            {
                "feature": [9.0, 10.0, 11.0, 12.0],
                "category": ["a", "b", "z", "c"],
            }
        ),
        "y_test": pd.Series([0, 1, 0, 1]),
    }
