from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.experiments.model_v2_cost_sensitive_policy import (
    select_best_cost_sensitive_policy,
)
from ml.experiments.model_v2_model_family_comparison import (
    LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
    LIGHTGBM_POLICY_THRESHOLD,
    MODEL_FAMILY_MAX_ALERT_RATE,
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


def get_catboost_availability() -> dict[str, Any]:
    try:
        import catboost  # noqa: F401

        return {"available": True, "reason": "available"}
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}


def generate_catboost_tuning_candidates() -> list[dict[str, Any]]:
    """Return small, controlled CatBoost tuning candidates."""

    return [
        {
            "candidate": "catboost_default",
            "params": {},
        },
        {
            "candidate": "catboost_class_weights_moderate",
            "params": {"class_weights": [1.0, 3.0]},
        },
        {
            "candidate": "catboost_auto_class_weights_balanced",
            "params": {"auto_class_weights": "Balanced"},
        },
        {
            "candidate": "catboost_depth_5_lr_003_iterations_700",
            "params": {"depth": 5, "learning_rate": 0.03, "iterations": 700},
        },
        {
            "candidate": "catboost_depth_7_lr_004_iterations_500",
            "params": {"depth": 7, "learning_rate": 0.04, "iterations": 500},
        },
    ]


def build_catboost_policy_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        if candidate["status"] != "trained":
            continue
        for threshold_row in candidate["validation_threshold_comparison"]:
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "model_family": "catboost",
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


def select_best_catboost_policy(
    policy_rows: list[dict[str, Any]],
    *,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any] | None:
    return select_best_cost_sensitive_policy(
        policy_rows,
        max_alert_rate=max_alert_rate,
    )


def run_model_v2_catboost_tuning_experiment(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
) -> dict[str, Any]:
    """
    Tune CatBoost candidates under the Model v2 feature pipeline.

    This experiment is non-mutating. It does not write artifacts, update
    thresholds, promote models, or change production inference behavior.
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

    benchmark = _train_lightgbm_benchmark(
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        X_test=X_test_v2,
        y_test=splits["y_test"],
        categorical_cols=categorical_cols,
    )
    availability = get_catboost_availability()
    catboost_candidates = [
        _train_or_skip_catboost_candidate(
            availability=availability,
            candidate_config=candidate_config,
            X_train=X_train_v2,
            y_train=splits["y_train"],
            X_val=X_val_v2,
            y_val=splits["y_val"],
            X_test=X_test_v2,
            y_test=splits["y_test"],
            categorical_cols=categorical_cols,
        )
        for candidate_config in generate_catboost_tuning_candidates()
    ]
    policy_rows = build_catboost_policy_rows(catboost_candidates)
    best_catboost_policy = select_best_catboost_policy(policy_rows)

    return {
        "experiment": "model_v2_catboost_tuning",
        "write_artifacts": False,
        "artifacts_written": False,
        "max_alert_rate": MODEL_FAMILY_MAX_ALERT_RATE,
        "feature_count": len(transformer.feature_names_),
        "catboost_availability": availability,
        "lightgbm_benchmark": benchmark,
        "catboost_candidates": catboost_candidates,
        "policy_rows": policy_rows,
        "best_catboost_policy": best_catboost_policy,
    }


def _train_lightgbm_benchmark(
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
    return {
        "candidate": "lightgbm_scale_pos_weight_5.0000",
        "model_family": "lightgbm",
        "scale_pos_weight": LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
        "threshold": LIGHTGBM_POLICY_THRESHOLD,
        "validation_metrics": evaluate_predictions_v2(
            y_true=y_val,
            y_proba=val_proba,
            threshold=LIGHTGBM_POLICY_THRESHOLD,
        ),
        "test_metrics": evaluate_predictions_v2(
            y_true=y_test,
            y_proba=test_proba,
            threshold=LIGHTGBM_POLICY_THRESHOLD,
        ),
    }


def _train_or_skip_catboost_candidate(
    *,
    availability: dict[str, Any],
    candidate_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    if not availability["available"]:
        return {
            "candidate": candidate_config["candidate"],
            "model_family": "catboost",
            "status": "skipped",
            "skip_reason": availability["reason"],
            "params": candidate_config["params"],
        }

    gpu_params = _catboost_params(candidate_config["params"], task_type="GPU")
    try:
        return _fit_catboost_candidate(
            candidate=candidate_config["candidate"],
            params=gpu_params,
            execution_device="GPU",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
        )
    except Exception as exc:
        cpu_params = _catboost_params(candidate_config["params"], task_type="CPU")
        result = _fit_catboost_candidate(
            candidate=candidate_config["candidate"],
            params=cpu_params,
            execution_device="CPU fallback",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
        )
        result["gpu_error"] = str(exc)
        return result


def _catboost_params(overrides: dict[str, Any], *, task_type: str) -> dict[str, Any]:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False,
        "task_type": task_type,
    }
    if task_type == "GPU":
        params["devices"] = "0"
    params.update(overrides)
    return params


def _fit_catboost_candidate(
    *,
    candidate: str,
    params: dict[str, Any],
    execution_device: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_cols: list[str],
) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    train_pool = _build_catboost_pool(X_train, y_train, categorical_cols)
    val_pool = _build_catboost_pool(X_val, y_val, categorical_cols)
    test_pool = _build_catboost_pool(X_test, y_test, categorical_cols)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_proba = model.predict_proba(val_pool)[:, 1]
    test_proba = model.predict_proba(test_pool)[:, 1]

    return _build_catboost_candidate_result(
        candidate=candidate,
        params=params,
        execution_device=execution_device,
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


def _build_catboost_candidate_result(
    *,
    candidate: str,
    params: dict[str, Any],
    execution_device: str,
    y_val: pd.Series,
    val_proba: Any,
    y_test: pd.Series,
    test_proba: Any,
) -> dict[str, Any]:
    validation_threshold_comparison = evaluate_model_v2_thresholds(
        y_true=y_val,
        y_proba=np.asarray(val_proba),
    ).to_dict(orient="records")
    policy = select_best_catboost_policy(
        [
            {
                "candidate": candidate,
                "model_family": "catboost",
                **row,
            }
            for row in validation_threshold_comparison
        ]
    )
    selected_threshold = policy["threshold"] if policy is not None else 0.10
    return {
        "candidate": candidate,
        "model_family": "catboost",
        "status": "trained",
        "execution_device": execution_device,
        "params": params,
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
        "validation_threshold_comparison": validation_threshold_comparison,
        "test_threshold_comparison": evaluate_model_v2_thresholds(
            y_true=y_test,
            y_proba=np.asarray(test_proba),
        ).to_dict(orient="records"),
    }
