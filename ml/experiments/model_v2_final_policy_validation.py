from __future__ import annotations

from typing import Any

import pandas as pd

from ml.experiments.model_v2_catboost_tuning import (
    _build_catboost_pool,
    _catboost_params,
    get_catboost_availability,
)
from ml.experiments.model_v2_model_family_comparison import MODEL_FAMILY_MAX_ALERT_RATE
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


PROMOTION_PROMOTE_CANDIDATE = "promote_candidate"
PROMOTION_NEEDS_MORE_VALIDATION = "needs_more_validation"
PROMOTION_DO_NOT_PROMOTE = "do_not_promote"


def get_final_policy_candidates() -> list[dict[str, Any]]:
    """Return fixed Model v2 candidate policies for final validation."""

    return [
        {
            "candidate": "lightgbm_constrained",
            "model_family": "lightgbm",
            "scale_pos_weight": 1.0,
            "threshold": 0.10,
            "params": {},
        },
        {
            "candidate": "lightgbm_high_recall",
            "model_family": "lightgbm",
            "scale_pos_weight": 5.0,
            "threshold": 0.20,
            "params": {},
        },
        {
            "candidate": "catboost_default",
            "model_family": "catboost",
            "scale_pos_weight": None,
            "threshold": 0.10,
            "params": {},
        },
        {
            "candidate": "catboost_auto_class_weights_balanced",
            "model_family": "catboost",
            "scale_pos_weight": None,
            "threshold": 0.70,
            "params": {"auto_class_weights": "Balanced"},
        },
    ]


def run_model_v2_final_policy_validation(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any]:
    """
    Validate final Model v2 candidate policies in memory only.

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

    catboost_availability = get_catboost_availability()
    candidates = []
    for policy in get_final_policy_candidates():
        if policy["model_family"] == "lightgbm":
            candidates.append(
                _train_lightgbm_policy(
                    policy=policy,
                    X_train=X_train_v2,
                    y_train=splits["y_train"],
                    X_val=X_val_v2,
                    y_val=splits["y_val"],
                    X_test=X_test_v2,
                    y_test=splits["y_test"],
                    categorical_cols=categorical_cols,
                )
            )
        elif policy["model_family"] == "catboost":
            candidates.append(
                _train_or_skip_catboost_policy(
                    availability=catboost_availability,
                    policy=policy,
                    X_train=X_train_v2,
                    y_train=splits["y_train"],
                    X_val=X_val_v2,
                    y_val=splits["y_val"],
                    X_test=X_test_v2,
                    y_test=splits["y_test"],
                    categorical_cols=categorical_cols,
                )
            )

    decision = select_final_policy_decision(
        candidates,
        max_alert_rate=max_alert_rate,
    )
    return {
        "experiment": "model_v2_final_policy_validation",
        "write_artifacts": False,
        "artifacts_written": False,
        "max_alert_rate": max_alert_rate,
        "feature_count": len(transformer.feature_names_),
        "categorical_feature_count": len(categorical_cols),
        "catboost_availability": catboost_availability,
        "candidates": candidates,
        "decision": decision,
    }


def select_final_policy_decision(
    candidates: list[dict[str, Any]],
    *,
    max_alert_rate: float = MODEL_FAMILY_MAX_ALERT_RATE,
) -> dict[str, Any]:
    """Select a conservative final policy candidate."""

    trained_candidates = [
        candidate for candidate in candidates if candidate.get("status") == "trained"
    ]
    eligible = [
        candidate
        for candidate in trained_candidates
        if candidate["validation_metrics"]["alert_rate"] <= max_alert_rate
        and candidate["test_metrics"]["alert_rate"] <= max_alert_rate
    ]
    if not eligible:
        return {
            "recommended_candidate": None,
            "promotion_recommendation": PROMOTION_DO_NOT_PROMOTE,
            "reason": (
                "No trained candidate satisfied alert_rate <= "
                f"{max_alert_rate:.2f} on both validation and test."
            ),
            "risks": [
                "Alert volume may exceed the business review-capacity constraint.",
                "Candidate performance should not be promoted without a stable "
                "validation and test operating point.",
            ],
        }

    selected = sorted(
        eligible,
        key=lambda candidate: (
            candidate["test_metrics"]["recall"],
            candidate["test_metrics"]["precision"],
            candidate["test_metrics"]["f1_score"],
        ),
        reverse=True,
    )[0]
    return {
        "recommended_candidate": selected["candidate"],
        "promotion_recommendation": PROMOTION_PROMOTE_CANDIDATE,
        "reason": (
            "Selected the eligible candidate with highest test recall, then "
            "highest test precision, then highest test F1-score."
        ),
        "risks": [
            "Promotion still requires artifact creation, reproducibility checks, "
            "and API integration in separate reviewed changes.",
            "Production monitoring and rollback criteria must be defined before "
            "serving Model v2 traffic.",
        ],
        "selected_policy": {
            "candidate": selected["candidate"],
            "model_family": selected["model_family"],
            "scale_pos_weight": selected.get("scale_pos_weight"),
            "threshold": selected["threshold"],
            "validation_alert_rate": selected["validation_metrics"]["alert_rate"],
            "test_alert_rate": selected["test_metrics"]["alert_rate"],
            "test_recall": selected["test_metrics"]["recall"],
            "test_precision": selected["test_metrics"]["precision"],
            "test_f1_score": selected["test_metrics"]["f1_score"],
        },
    }


def _train_lightgbm_policy(
    *,
    policy: dict[str, Any],
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
        scale_pos_weight=policy["scale_pos_weight"],
    )
    test_proba = model.predict_proba(X_test)[:, 1]
    return _build_trained_policy_result(
        policy=policy,
        status="trained",
        execution_device="CPU",
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
    )


def _train_or_skip_catboost_policy(
    *,
    availability: dict[str, Any],
    policy: dict[str, Any],
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
            **_policy_metadata(policy),
            "status": "skipped",
            "execution_device": "skipped",
            "skip_reason": availability["reason"],
        }

    gpu_params = _catboost_params(policy["params"], task_type="GPU")
    try:
        return _fit_catboost_policy(
            policy=policy,
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
        cpu_params = _catboost_params(policy["params"], task_type="CPU")
        result = _fit_catboost_policy(
            policy=policy,
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


def _fit_catboost_policy(
    *,
    policy: dict[str, Any],
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
    return _build_trained_policy_result(
        policy=policy,
        status="trained",
        execution_device=execution_device,
        params=params,
        y_val=y_val,
        val_proba=val_proba,
        y_test=y_test,
        test_proba=test_proba,
    )


def _build_trained_policy_result(
    *,
    policy: dict[str, Any],
    status: str,
    execution_device: str,
    y_val: pd.Series,
    val_proba: Any,
    y_test: pd.Series,
    test_proba: Any,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **_policy_metadata(policy),
        "status": status,
        "execution_device": execution_device,
        "params": params if params is not None else policy["params"],
        "validation_metrics": evaluate_predictions_v2(
            y_true=y_val,
            y_proba=val_proba,
            threshold=policy["threshold"],
        ),
        "test_metrics": evaluate_predictions_v2(
            y_true=y_test,
            y_proba=test_proba,
            threshold=policy["threshold"],
        ),
    }


def _policy_metadata(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": policy["candidate"],
        "model_family": policy["model_family"],
        "scale_pos_weight": policy.get("scale_pos_weight"),
        "threshold": policy["threshold"],
    }
