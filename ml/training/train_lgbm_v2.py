from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def train_lightgbm_v2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: list[str],
    *,
    scale_pos_weight: float | None = None,
) -> tuple[LGBMClassifier, Any]:
    """
    Train a Model v2 LightGBM classifier.

    Default behavior matches the baseline v2 LightGBM configuration. The
    optional ``scale_pos_weight`` parameter is used only when explicitly passed
    for recall-improvement experiments.
    """

    if X_train.empty or X_val.empty:
        raise ValueError("Training or validation feature set is empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train size mismatch.")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val size mismatch.")

    model_params = {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "random_state": 42,
        "n_jobs": -1,
    }
    if scale_pos_weight is not None:
        model_params["scale_pos_weight"] = scale_pos_weight

    model = LGBMClassifier(**model_params)

    missing_cols: list[str] = []
    if categorical_cols:
        missing_cols = [col for col in categorical_cols if col not in X_train.columns]
    if missing_cols:
        raise ValueError(f"Categorical columns missing in training data: {missing_cols}")

    if categorical_cols:
        for col in categorical_cols:
            if X_train[col].dtype.name != "category":
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[
            early_stopping(50),
            log_evaluation(0),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("Best iteration: %s", model.best_iteration_)
    logger.info("Best validation AUC: %s", model.best_score_["valid_0"]["auc"])

    proba = model.predict_proba(X_val)
    if proba.shape[1] != 2:
        raise ValueError("Model did not return binary class probabilities.")

    return model, proba[:, 1]
