from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.experiments.model_v2_cost_sensitive_policy import (
    select_best_cost_sensitive_policy,
)
from ml.pipelines.training_pipeline_v2 import (
    DEFAULT_IDENTITY_PATH,
    DEFAULT_TRANSACTION_PATH,
    align_categorical_features_for_lightgbm,
    evaluate_predictions_v2,
    load_transaction_identity_dataset,
    prepare_time_based_train_val_test_split,
    validate_feature_columns_match,
)
from ml.training.feature_engineering_v2 import FeatureEngineeringV2
from ml.training.train_lgbm_v2 import train_lightgbm_v2
from ml.utils.threshold_evaluation_v2 import evaluate_model_v2_thresholds


LIGHTGBM_POLICY_SCALE_POS_WEIGHT = 5.0
LIGHTGBM_POLICY_THRESHOLD = 0.20
MODEL_FAMILY_MAX_ALERT_RATE = 0.05


def get_optional_model_family_availability() -> dict[str, dict[str, Any]]:
    """Return optional model-family availability without importing at module load."""

    availability = {
        "lightgbm": {"available": True, "reason": "available"},
        "xgboost": {"available": False, "reason": "xgboost is not installed"},
        "catboost": {"available": False, "reason": "catboost is not installed"},
    }

    try:
        import xgboost  # noqa: F401

        availability["xgboost"] = {"available": True, "reason": "available"}
    except ImportError as exc:
        availability["xgboost"] = {"available": False, "reason": str(exc)}

    try:
        import catboost  # noqa: F401

        availability["catboost"] = {"available": True, "reason": "available"}
    except ImportError as exc:
        availability["catboost"] = {"available": False, "reason": str(exc)}

    return availability


def build_model_family_policy_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten candidate threshold tables into policy rows."""

    rows = []
    for candidate in candidates:
        if candidate["status"] != "trained":
            continue
        for threshold_row in candidate["validation_threshold_comparison"]:
            rows.append(
                {
                    "model_family": candidate["model_family"],
                    "candidate": candidate["candidate"],
                    "scale_pos_weight": candidate.get("scale_pos_weight"),
                    "threshold": threshold_row["threshold"],
                    "precision": threshold_row["precision"],
                    "recall": threshold_row["recall"],
                    "f1_score": threshold_row["f1_score"],
                    "alert_rate": threshold_row["alert_rate"],
                    "false_positives": threshold_row["false_positives"],
                    "false_negatives": threshold_row["false_negatives"],
                    "true_positives": threshold_row["true_positives"],
                    "true_negatives": threshold_row["true_negatives"],
                }
            )
    return rows


def select_best_model_family_policy(
    policy_rows: list[dict[str, Any]],
    *,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any] | None:
    """Select best model-family policy under the shared alert-rate constraint."""

    return select_best_cost_sensitive_policy(
        policy_rows,
        max_alert_rate=max_alert_rate,
    )


def run_model_v2_model_family_comparison_experiment(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
) -> dict[str, Any]:
    """
    Compare Model v2 LightGBM, XGBoost, and CatBoost candidates in memory.

    Optional model families are skipped when their dependencies are missing.
    This workflow does not write artifacts, update thresholds, promote models,
    or change production inference behavior.
    """

    merged = load_transaction_identity_dataset(
        transaction_path=transaction_path,
        identity_path=identity_path,
    )
    splits = prepare_time_based_train_val_test_split(merged)

    transformer = FeatureEngineeringV2()
    X_train_v2 = transformer.fit_transform(splits["X_train"])
    X_val_v2 = transformer.transform(splits["X_val"])
    X_test_v2 = transformer.transform(splits["X_test"])
    validate_feature_columns_match(X_train_v2, X_val_v2, X_test_v2)

    categorical_cols = [
        col for col in transformer.categorical_columns_ if col in X_train_v2.columns
    ]
    X_train_v2, X_val_v2, X_test_v2 = align_categorical_features_for_lightgbm(
        X_train=X_train_v2,
        X_val=X_val_v2,
        X_test=X_test_v2,
        categorical_cols=categorical_cols,
    )

    availability = get_optional_model_family_availability()
    candidates = [
        _train_lightgbm_candidate(
            X_train=X_train_v2,
            y_train=splits["y_train"],
            X_val=X_val_v2,
            y_val=splits["y_val"],
            X_test=X_test_v2,
            y_test=splits["y_test"],
            categorical_cols=categorical_cols,
        )
    ]
    candidates.append(
        _train_optional_xgboost_candidate(
            availability=availability["xgboost"],
            X_train=X_train_v2,
            y_train=splits["y_train"],
            X_val=X_val_v2,
            y_val=splits["y_val"],
            X_test=X_test_v2,
            y_test=splits["y_test"],
        )
    )
    candidates.append(
        _train_optional_catboost_candidate(
            availability=availability["catboost"],
            X_train=X_train_v2,
            y_train=splits["y_train"],
            X_val=X_val_v2,
            y_val=splits["y_val"],
            X_test=X_test_v2,
            y_test=splits["y_test"],
            categorical_cols=categorical_cols,
        )
    )

    policy_rows = build_model_family_policy_rows(candidates)
    best_policy = select_best_model_family_policy(policy_rows)

    return {
        "experiment": "model_v2_model_family_comparison",
        "write_artifacts": False,
        "artifacts_written": False,
        "max_alert_rate": MODEL_FAMILY_MAX_ALERT_RATE,
        "feature_count": len(transformer.feature_names_),
        "availability": availability,
        "benchmark": {
            "model_family": "lightgbm",
            "scale_pos_weight": LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
            "threshold": LIGHTGBM_POLICY_THRESHOLD,
        },
        "candidates": candidates,
        "policy_rows": policy_rows,
        "best_policy": best_policy,
    }


def _train_lightgbm_candidate(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    model, val_proba = train_lightgbm_v2(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=categorical_cols,
        scale_pos_weight=LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
    )
    test_proba = model.predict_proba(X_test)[:, 1]
    return _build_trained_candidate(
        model_family="lightgbm",
        candidate="lightgbm_scale_pos_weight_5.0000",
        scale_pos_weight=LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
        benchmark_threshold=LIGHTGBM_POLICY_THRESHOLD,
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
    )


def _train_optional_xgboost_candidate(
    *,
    availability: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    if not availability["available"]:
        return _skipped_candidate("xgboost", availability["reason"])

    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="auc",
        enable_categorical=True,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    return _build_trained_candidate(
        model_family="xgboost",
        candidate="xgboost_default_v2",
        scale_pos_weight=None,
        benchmark_threshold=None,
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
    )


def _train_optional_catboost_candidate(
    *,
    availability: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    if not availability["available"]:
        return _skipped_candidate("catboost", availability["reason"])

    from catboost import CatBoostClassifier

    train_pool = _build_catboost_pool(X_train, y_train, categorical_cols)
    val_pool = _build_catboost_pool(X_val, y_val, categorical_cols)
    test_pool = _build_catboost_pool(X_test, y_test, categorical_cols)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_proba = model.predict_proba(val_pool)[:, 1]
    test_proba = model.predict_proba(test_pool)[:, 1]
    return _build_trained_candidate(
        model_family="catboost",
        candidate="catboost_default_v2",
        scale_pos_weight=None,
        benchmark_threshold=None,
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
    )


def _build_catboost_pool(X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]):
    from catboost import Pool

    X_pool = X.copy()
    for col in categorical_cols:
        X_pool[col] = X_pool[col].astype(str)
    cat_features = [X_pool.columns.get_loc(col) for col in categorical_cols]
    return Pool(X_pool, label=y, cat_features=cat_features)


def _build_trained_candidate(
    *,
    model_family: str,
    candidate: str,
    scale_pos_weight: float | None,
    benchmark_threshold: float | None,
    y_val: pd.Series,
    val_proba: Any,
    y_test: pd.Series,
    test_proba: Any,
) -> dict[str, Any]:
    threshold_comparison = evaluate_thresholds_for_policy(y_val, val_proba)
    selected_policy = select_best_model_family_policy(
        [
            {
                "model_family": model_family,
                "candidate": candidate,
                "scale_pos_weight": scale_pos_weight,
                **row,
            }
            for row in threshold_comparison
        ]
    )
    if benchmark_threshold is not None:
        selected_threshold = benchmark_threshold
    elif selected_policy is not None:
        selected_threshold = selected_policy["threshold"]
    else:
        selected_threshold = LIGHTGBM_POLICY_THRESHOLD

    return {
        "model_family": model_family,
        "candidate": candidate,
        "status": "trained",
        "scale_pos_weight": scale_pos_weight,
        "benchmark_threshold": benchmark_threshold,
        "selected_threshold": selected_threshold,
        "validation_metrics": evaluate_predictions_v2(
            y_true=y_val,
            y_proba=val_proba,
            threshold=selected_threshold,
        ),
        "test_metrics": evaluate_predictions_v2(
            y_true=y_test,
            y_proba=test_proba,
            threshold=selected_threshold,
        ),
        "validation_threshold_comparison": threshold_comparison,
        "test_threshold_comparison": evaluate_thresholds_for_policy(y_test, test_proba),
    }


def evaluate_thresholds_for_policy(y_true: pd.Series, y_proba: Any) -> list[dict[str, Any]]:
    from ml.utils.threshold_evaluation_v2 import evaluate_model_v2_thresholds

    return evaluate_model_v2_thresholds(y_true=y_true, y_proba=np.asarray(y_proba)).to_dict(
        orient="records"
    )


def _skipped_candidate(model_family: str, reason: str) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "candidate": f"{model_family}_default_v2",
        "status": "skipped",
        "skip_reason": reason,
    }
