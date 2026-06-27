import json
import zipfile
from pathlib import Path

import joblib
import pytest

from scripts import validate_model_v2_catboost_artifacts as validation_script


def test_model_v2_catboost_artifact_validation_succeeds(tmp_path):
    artifact_zip = make_fake_artifact_zip(tmp_path)

    result = validation_script.validate_model_v2_catboost_artifacts(artifact_zip)

    assert result["feature_count"] == validation_script.EXPECTED_FEATURE_COUNT
    assert result["threshold"] == validation_script.EXPECTED_THRESHOLD
    assert result["model_version"] == "v2"
    assert result["model_family"] == "catboost"
    assert set(result["validated_files"]) == validation_script.REQUIRED_ARTIFACT_FILES


def test_model_v2_catboost_artifact_validation_rejects_missing_zip(tmp_path):
    missing_zip = tmp_path / "missing_model_v2_catboost_artifacts.zip"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validation_script.validate_model_v2_catboost_artifacts(missing_zip)


def test_model_v2_catboost_artifact_validation_rejects_missing_file_inside_zip(
    tmp_path,
):
    artifact_zip = make_fake_artifact_zip(
        tmp_path,
        omitted_files={"threshold_v2.json"},
    )

    with pytest.raises(ValueError, match="missing required files"):
        validation_script.validate_model_v2_catboost_artifacts(artifact_zip)


def test_model_v2_catboost_artifact_validation_rejects_wrong_feature_count(tmp_path):
    artifact_zip = make_fake_artifact_zip(
        tmp_path,
        overrides={"feature_columns_v2.json": ["feature_a", "feature_b"]},
    )

    with pytest.raises(ValueError, match="exactly 831"):
        validation_script.validate_model_v2_catboost_artifacts(artifact_zip)


def test_model_v2_catboost_artifact_validation_rejects_wrong_threshold(tmp_path):
    artifact_zip = make_fake_artifact_zip(
        tmp_path,
        overrides={"threshold_v2.json": {"threshold": 0.20}},
    )

    with pytest.raises(ValueError, match="threshold 0.10"):
        validation_script.validate_model_v2_catboost_artifacts(artifact_zip)


def test_model_v2_catboost_artifact_validation_rejects_wrong_metadata(tmp_path):
    metadata = make_valid_metadata()
    metadata["model_family"] = "lightgbm"
    artifact_zip = make_fake_artifact_zip(
        tmp_path,
        overrides={"metadata_v2.json": metadata},
    )

    with pytest.raises(ValueError, match="model_family"):
        validation_script.validate_model_v2_catboost_artifacts(artifact_zip)


def test_model_v2_catboost_artifact_validation_rejects_invalid_joblib_file(tmp_path):
    artifact_zip = make_fake_artifact_zip(
        tmp_path,
        invalid_joblib_files={"fraud_catboost_v2.joblib"},
    )

    with pytest.raises(ValueError, match="Unable to load fraud_catboost_v2.joblib"):
        validation_script.validate_model_v2_catboost_artifacts(artifact_zip)


def make_fake_artifact_zip(
    tmp_path: Path,
    *,
    omitted_files: set[str] | None = None,
    overrides: dict[str, object] | None = None,
    invalid_joblib_files: set[str] | None = None,
) -> Path:
    omitted_files = omitted_files or set()
    overrides = overrides or {}
    invalid_joblib_files = invalid_joblib_files or set()

    artifact_zip = tmp_path / "model_v2_catboost_artifacts.zip"
    with zipfile.ZipFile(artifact_zip, mode="w") as archive:
        for filename, payload in make_valid_json_payloads().items():
            if filename in omitted_files:
                continue
            archive.writestr(
                filename,
                json.dumps(overrides.get(filename, payload)),
            )

        for filename in (
            "fraud_catboost_v2.joblib",
            "feature_transformer_v2.joblib",
        ):
            if filename in omitted_files:
                continue
            if filename in invalid_joblib_files:
                archive.writestr(filename, b"not a joblib artifact")
            else:
                archive.write(make_temp_joblib(tmp_path, filename), arcname=filename)

    return artifact_zip


def make_valid_json_payloads() -> dict[str, object]:
    return {
        "feature_columns_v2.json": [
            f"feature_{idx}"
            for idx in range(validation_script.EXPECTED_FEATURE_COUNT)
        ],
        "threshold_v2.json": {"threshold": validation_script.EXPECTED_THRESHOLD},
        "metadata_v2.json": make_valid_metadata(),
        "model_v2_evaluation_report.json": {
            "validation_metrics": {"recall": 0.90},
            "test_metrics": {"recall": 0.88},
        },
    }


def make_valid_metadata() -> dict[str, object]:
    return {
        "model_version": "v2",
        "model_family": "catboost",
        "n_features": validation_script.EXPECTED_FEATURE_COUNT,
        "threshold": validation_script.EXPECTED_THRESHOLD,
    }


def make_temp_joblib(tmp_path: Path, filename: str) -> Path:
    joblib_path = tmp_path / filename
    joblib.dump({"artifact": filename}, joblib_path)
    return joblib_path
