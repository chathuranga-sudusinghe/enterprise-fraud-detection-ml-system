from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.experiments.model_v2_model_family_comparison import (
    LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
    LIGHTGBM_POLICY_THRESHOLD,
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


DEFAULT_GROUP_COLUMNS = [
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "TransactionAmt_band",
    "transaction_hour_band",
    "identity_missing_count_band",
]


def run_model_v2_false_negative_analysis(
    *,
    transaction_path=DEFAULT_TRANSACTION_PATH,
    identity_path=DEFAULT_IDENTITY_PATH,
    threshold: float = LIGHTGBM_POLICY_THRESHOLD,
    scale_pos_weight: float = LIGHTGBM_POLICY_SCALE_POS_WEIGHT,
) -> dict[str, Any]:
    """
    Train the selected Model v2 LightGBM policy and analyze missed fraud cases.

    This workflow is intentionally non-mutating. It trains in memory only and
    does not write model artifacts, update thresholds, promote models, or change
    production inference behavior.
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
    model, val_proba = train_lightgbm_v2(
        X_train=X_train_v2,
        y_train=splits["y_train"],
        X_val=X_val_v2,
        y_val=splits["y_val"],
        categorical_cols=categorical_cols,
        scale_pos_weight=scale_pos_weight,
    )
    test_proba = model.predict_proba(X_test_v2)[:, 1]

    validation_analysis = analyze_false_negatives(
        X=splits["X_val"],
        y_true=splits["y_val"],
        y_proba=val_proba,
        threshold=threshold,
    )
    test_analysis = analyze_false_negatives(
        X=splits["X_test"],
        y_true=splits["y_test"],
        y_proba=test_proba,
        threshold=threshold,
    )

    return {
        "experiment": "model_v2_false_negative_analysis",
        "write_artifacts": False,
        "artifacts_written": False,
        "model_family": "lightgbm",
        "scale_pos_weight": scale_pos_weight,
        "threshold": threshold,
        "feature_count": len(transformer.feature_names_),
        "categorical_feature_count": len(categorical_cols),
        "validation_metrics": evaluate_predictions_v2(
            y_true=splits["y_val"],
            y_proba=val_proba,
            threshold=threshold,
        ),
        "test_metrics": evaluate_predictions_v2(
            y_true=splits["y_test"],
            y_proba=test_proba,
            threshold=threshold,
        ),
        "validation_analysis": validation_analysis,
        "test_analysis": test_analysis,
    }


def analyze_false_negatives(
    *,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_proba: Any,
    threshold: float,
    group_columns: list[str] | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build split-level false-negative summaries from raw split rows."""

    outcome_frame = build_outcome_frame(
        X=X,
        y_true=y_true,
        y_proba=y_proba,
        threshold=threshold,
    )
    outcome_frame = add_false_negative_analysis_columns(outcome_frame)
    group_summary = summarize_false_negative_groups(
        outcome_frame,
        group_columns=group_columns or DEFAULT_GROUP_COLUMNS,
        top_n=top_n,
    )
    false_negatives = outcome_frame[outcome_frame["_is_false_negative"]].copy()
    true_positives = outcome_frame[outcome_frame["_is_true_positive"]].copy()
    return {
        "summary": summarize_false_negative_counts(outcome_frame),
        "top_false_negative_groups": group_summary,
        "false_negative_rows": false_negatives,
        "true_positive_rows": true_positives,
    }


def build_outcome_frame(
    *,
    X: pd.DataFrame,
    y_true: pd.Series,
    y_proba: Any,
    threshold: float,
) -> pd.DataFrame:
    """Attach predictions and outcome flags to raw split rows."""

    if len(X) != len(y_true):
        raise ValueError("X and y_true must have matching row counts.")

    y_proba_array = np.asarray(y_proba)
    if len(y_proba_array) != len(y_true):
        raise ValueError("y_proba and y_true must have matching row counts.")

    frame = X.reset_index(drop=True).copy()
    y_true_values = pd.Series(y_true).reset_index(drop=True).astype(int)
    y_pred_values = (y_proba_array >= threshold).astype(int)

    frame["_y_true"] = y_true_values
    frame["_y_proba"] = y_proba_array
    frame["_y_pred"] = y_pred_values
    frame["_is_false_negative"] = (y_true_values == 1) & (y_pred_values == 0)
    frame["_is_true_positive"] = (y_true_values == 1) & (y_pred_values == 1)
    frame["_is_non_fraud"] = y_true_values == 0
    return frame


def summarize_false_negative_counts(outcome_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize missed-fraud volume for one split."""

    fraud_count = int((outcome_frame["_y_true"] == 1).sum())
    false_negative_count = int(outcome_frame["_is_false_negative"].sum())
    return {
        "row_count": int(len(outcome_frame)),
        "fraud_count": fraud_count,
        "false_negative_count": false_negative_count,
        "true_positive_count": int(outcome_frame["_is_true_positive"].sum()),
        "missed_fraud_rate": (
            float(false_negative_count / fraud_count) if fraud_count else 0.0
        ),
    }


def add_false_negative_analysis_columns(outcome_frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic bands used for group-level missed-fraud analysis."""

    frame = outcome_frame.copy()
    if "TransactionAmt" in frame.columns:
        frame["TransactionAmt_band"] = pd.cut(
            pd.to_numeric(frame["TransactionAmt"], errors="coerce"),
            bins=[-np.inf, 25, 50, 100, 250, 500, np.inf],
            labels=["<=25", "25-50", "50-100", "100-250", "250-500", ">500"],
        ).astype("object")

    if "TransactionDT" in frame.columns:
        hour = (pd.to_numeric(frame["TransactionDT"], errors="coerce") // 3600) % 24
        frame["transaction_hour_band"] = pd.cut(
            hour,
            bins=[-1, 5, 11, 17, 23],
            labels=["overnight", "morning", "afternoon", "evening"],
        ).astype("object")

    identity_columns = [col for col in frame.columns if col.startswith("id_")]
    if identity_columns:
        missing_counts = frame[identity_columns].isna().sum(axis=1)
        frame["identity_missing_count"] = missing_counts
        frame["identity_missing_count_band"] = pd.cut(
            missing_counts,
            bins=[-1, 0, 5, 20, np.inf],
            labels=["0", "1-5", "6-20", "21+"],
        ).astype("object")

    return frame


def summarize_false_negative_groups(
    outcome_frame: pd.DataFrame,
    *,
    group_columns: list[str],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Summarize false negatives against fraud and non-fraud counts by group."""

    rows: list[dict[str, Any]] = []
    total_false_negatives = int(outcome_frame["_is_false_negative"].sum())
    for group_column in group_columns:
        if group_column not in outcome_frame.columns:
            continue
        group_values = outcome_frame[group_column].astype("object").where(
            outcome_frame[group_column].notna(),
            "__MISSING__",
        )
        grouped = outcome_frame.assign(_group_value=group_values).groupby(
            "_group_value",
            dropna=False,
        )
        for group_value, group in grouped:
            false_negatives = int(group["_is_false_negative"].sum())
            if false_negatives == 0:
                continue
            fraud_count = int((group["_y_true"] == 1).sum())
            total_count = int(len(group))
            rows.append(
                {
                    "group_column": group_column,
                    "group_value": str(group_value),
                    "false_negatives": false_negatives,
                    "true_positives": int(group["_is_true_positive"].sum()),
                    "fraud_count": fraud_count,
                    "non_fraud_count": int((group["_y_true"] == 0).sum()),
                    "total_count": total_count,
                    "missed_fraud_rate": (
                        float(false_negatives / fraud_count) if fraud_count else 0.0
                    ),
                    "false_negative_share": (
                        float(false_negatives / total_false_negatives)
                        if total_false_negatives
                        else 0.0
                    ),
                    "fraud_rate": float(fraud_count / total_count) if total_count else 0.0,
                }
            )

    return sorted(
        rows,
        key=lambda row: (
            row["false_negatives"],
            row["missed_fraud_rate"],
            row["false_negative_share"],
        ),
        reverse=True,
    )[:top_n]
