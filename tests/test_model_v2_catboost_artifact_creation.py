from pathlib import Path

import pandas as pd
import pytest

from scripts import create_model_v2_catboost_artifacts as artifact_script


def test_catboost_artifact_creation_dry_run_does_not_write(monkeypatch, tmp_path):
    install_artifact_creation_monkeypatches(monkeypatch)
    artifact_paths = make_temp_artifact_paths(tmp_path)

    def fail_write(*args, **kwargs):
        raise AssertionError("dry-run must not write artifacts")

    monkeypatch.setattr(artifact_script, "write_catboost_v2_artifacts", fail_write)

    summary = artifact_script.create_model_v2_catboost_artifacts(
        artifact_paths=artifact_paths,
        write_artifacts=False,
    )

    assert summary["would_write_artifacts"] is False
    assert summary["artifacts_written"] is False
    assert summary["feature_count"] == artifact_script.EXPECTED_MODEL_V2_FEATURE_COUNT
    assert summary["threshold"] == artifact_script.CATBOOST_V2_THRESHOLD
    assert not any(path.exists() for path in artifact_paths.values())


def test_catboost_artifact_creation_rejects_existing_v2_without_overwrite(tmp_path):
    artifact_paths = make_temp_artifact_paths(tmp_path)
    artifact_paths["threshold"].write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite-v2"):
        artifact_script.validate_catboost_v2_artifact_paths(
            artifact_paths,
            write_artifacts=True,
            overwrite_v2=False,
        )


def test_catboost_artifact_creation_allows_existing_v2_with_overwrite(tmp_path):
    artifact_paths = make_temp_artifact_paths(tmp_path)
    artifact_paths["threshold"].write_text("{}", encoding="utf-8")

    artifact_script.validate_catboost_v2_artifact_paths(
        artifact_paths,
        write_artifacts=True,
        overwrite_v2=True,
    )


def test_catboost_artifact_creation_rejects_v1_artifact_names(tmp_path):
    artifact_paths = make_temp_artifact_paths(tmp_path)
    artifact_paths["model"] = tmp_path / "fraud_lgbm_v1.joblib"

    with pytest.raises(ValueError, match="v1 artifacts"):
        artifact_script.validate_catboost_v2_artifact_paths(
            artifact_paths,
            write_artifacts=False,
            overwrite_v2=False,
        )


def test_catboost_artifact_creation_rejects_non_v2_artifact_names(tmp_path):
    artifact_paths = make_temp_artifact_paths(tmp_path)
    artifact_paths["model"] = tmp_path / "fraud_catboost.joblib"

    with pytest.raises(ValueError, match="v2-specific"):
        artifact_script.validate_catboost_v2_artifact_paths(
            artifact_paths,
            write_artifacts=False,
            overwrite_v2=False,
        )


def test_catboost_artifact_creation_rejects_unexpected_threshold(
    monkeypatch,
    tmp_path,
):
    install_artifact_creation_monkeypatches(monkeypatch)

    with pytest.raises(ValueError, match="requires threshold"):
        artifact_script.create_model_v2_catboost_artifacts(
            artifact_paths=make_temp_artifact_paths(tmp_path),
            threshold=0.20,
        )


def test_catboost_artifact_creation_rejects_wrong_feature_count(
    monkeypatch,
    tmp_path,
):
    install_artifact_creation_monkeypatches(monkeypatch, feature_count=3)

    with pytest.raises(ValueError, match="Expected 831"):
        artifact_script.create_model_v2_catboost_artifacts(
            artifact_paths=make_temp_artifact_paths(tmp_path),
        )


def test_catboost_artifact_creation_writes_only_explicit_temp_v2_artifacts(
    monkeypatch,
    tmp_path,
):
    install_artifact_creation_monkeypatches(monkeypatch)
    artifact_paths = make_temp_artifact_paths(tmp_path)

    summary = artifact_script.create_model_v2_catboost_artifacts(
        artifact_paths=artifact_paths,
        write_artifacts=True,
        overwrite_v2=False,
    )

    assert summary["artifacts_written"] is True
    assert set(path.name for path in artifact_paths.values()) == {
        "fraud_catboost_v2.joblib",
        "feature_transformer_v2.joblib",
        "feature_columns_v2.json",
        "metadata_v2.json",
        "threshold_v2.json",
        "model_v2_evaluation_report.json",
    }
    assert all(path.exists() for path in artifact_paths.values())


def install_artifact_creation_monkeypatches(monkeypatch, *, feature_count=831):
    monkeypatch.setattr(
        artifact_script,
        "load_transaction_identity_dataset",
        lambda transaction_path, identity_path: pd.DataFrame({"placeholder": [1]}),
    )
    monkeypatch.setattr(
        artifact_script,
        "prepare_time_based_train_val_test_split",
        lambda merged: synthetic_splits(),
    )
    monkeypatch.setattr(
        artifact_script,
        "FeatureEngineeringV2",
        lambda: FakeTransformer(feature_count),
    )
    monkeypatch.setattr(
        artifact_script,
        "train_catboost_default_candidate",
        fake_train_catboost_default_candidate,
    )


def make_temp_artifact_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "model": tmp_path / "fraud_catboost_v2.joblib",
        "transformer": tmp_path / "feature_transformer_v2.joblib",
        "feature_columns": tmp_path / "feature_columns_v2.json",
        "metadata": tmp_path / "metadata_v2.json",
        "threshold": tmp_path / "threshold_v2.json",
        "evaluation_report": tmp_path / "model_v2_evaluation_report.json",
    }


def synthetic_splits():
    return {
        "X_train": pd.DataFrame({"raw": [1, 2, 3, 4]}),
        "y_train": pd.Series([0, 1, 0, 1]),
        "X_val": pd.DataFrame({"raw": [5, 6, 7, 8]}),
        "y_val": pd.Series([0, 1, 0, 1]),
        "X_test": pd.DataFrame({"raw": [9, 10, 11, 12]}),
        "y_test": pd.Series([0, 1, 0, 1]),
    }


class FakeTransformer:
    categorical_columns_ = []

    def __init__(self, feature_count):
        self.feature_names_ = [f"feature_{idx}" for idx in range(feature_count)]

    def fit_transform(self, X):
        return pd.DataFrame(
            {
                feature_name: [float(idx)] * len(X)
                for idx, feature_name in enumerate(self.feature_names_)
            }
        )

    def transform(self, X):
        return self.fit_transform(X)


class FakeModel:
    best_iteration_ = 11


def fake_train_catboost_default_candidate(**kwargs):
    return {
        "model": FakeModel(),
        "execution_device": "GPU",
        "params": {"task_type": "GPU", "devices": "0"},
        "val_proba": [0.05, 0.95, 0.20, 0.80],
        "test_proba": [0.10, 0.90, 0.30, 0.70],
    }
