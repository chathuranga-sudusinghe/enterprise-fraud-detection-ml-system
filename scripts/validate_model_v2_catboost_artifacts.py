from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_FEATURE_COUNT = 831
EXPECTED_THRESHOLD = 0.10

REQUIRED_ARTIFACT_FILES = {
    "feature_columns_v2.json",
    "feature_transformer_v2.joblib",
    "fraud_catboost_v2.joblib",
    "metadata_v2.json",
    "model_v2_evaluation_report.json",
    "threshold_v2.json",
}


def validate_model_v2_catboost_artifacts(artifact_zip: str | Path) -> dict[str, Any]:
    """Validate the reproducibility contract for a Model v2 CatBoost artifact zip."""

    zip_path = Path(artifact_zip)
    if not zip_path.exists():
        raise FileNotFoundError(f"Artifact zip does not exist: {zip_path}")
    if not zip_path.is_file():
        raise FileNotFoundError(f"Artifact zip is not a file: {zip_path}")

    with zipfile.ZipFile(zip_path) as artifact_archive:
        archive_names = set(artifact_archive.namelist())
        missing_files = sorted(REQUIRED_ARTIFACT_FILES - archive_names)
        if missing_files:
            raise ValueError(f"Artifact zip is missing required files: {missing_files}")

        feature_columns = _read_json(artifact_archive, "feature_columns_v2.json")
        if not isinstance(feature_columns, list):
            raise ValueError("feature_columns_v2.json must contain an ordered list")
        if len(feature_columns) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "feature_columns_v2.json must contain exactly "
                f"{EXPECTED_FEATURE_COUNT} ordered features; found {len(feature_columns)}"
            )

        threshold_payload = _read_json(artifact_archive, "threshold_v2.json")
        threshold = _require_number(threshold_payload, "threshold_v2.json", "threshold")
        _validate_expected_threshold(threshold, "threshold_v2.json")

        metadata = _read_json(artifact_archive, "metadata_v2.json")
        _validate_metadata(metadata)

        evaluation_report = _read_json(
            artifact_archive,
            "model_v2_evaluation_report.json",
        )
        _validate_evaluation_report(evaluation_report)

        _load_joblib_member(artifact_archive, "fraud_catboost_v2.joblib")
        _load_joblib_member(artifact_archive, "feature_transformer_v2.joblib")

    return {
        "artifact_zip": str(zip_path),
        "feature_count": len(feature_columns),
        "threshold": threshold,
        "model_version": metadata["model_version"],
        "model_family": metadata["model_family"],
        "validated_files": sorted(REQUIRED_ARTIFACT_FILES),
    }


def _read_json(artifact_archive: zipfile.ZipFile, member_name: str) -> Any:
    with artifact_archive.open(member_name) as member_file:
        return json.load(member_file)


def _require_number(payload: Any, source_name: str, field_name: str) -> float:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_name} must contain a JSON object")
    value = payload.get(field_name)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{source_name} must contain numeric {field_name}")
    return float(value)


def _validate_expected_threshold(value: float, source_name: str) -> None:
    if abs(value - EXPECTED_THRESHOLD) > 1e-12:
        raise ValueError(
            f"{source_name} must contain threshold {EXPECTED_THRESHOLD:.2f}; "
            f"found {value:.2f}"
        )


def _validate_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata_v2.json must contain a JSON object")

    expected_metadata = {
        "model_version": "v2",
        "model_family": "catboost",
        "n_features": EXPECTED_FEATURE_COUNT,
        "threshold": EXPECTED_THRESHOLD,
    }
    for field_name, expected_value in expected_metadata.items():
        if field_name not in metadata:
            raise ValueError(f"metadata_v2.json is missing {field_name}")
        actual_value = metadata[field_name]
        if isinstance(expected_value, float):
            if not isinstance(actual_value, (int, float)):
                raise ValueError(f"metadata_v2.json {field_name} must be numeric")
            if abs(float(actual_value) - expected_value) > 1e-12:
                raise ValueError(
                    f"metadata_v2.json {field_name} must be "
                    f"{expected_value:.2f}; found {float(actual_value):.2f}"
                )
        elif actual_value != expected_value:
            raise ValueError(
                f"metadata_v2.json {field_name} must be "
                f"{expected_value!r}; found {actual_value!r}"
            )


def _validate_evaluation_report(evaluation_report: Any) -> None:
    if not isinstance(evaluation_report, dict):
        raise ValueError("model_v2_evaluation_report.json must contain a JSON object")
    for field_name in ("validation_metrics", "test_metrics"):
        if field_name not in evaluation_report:
            raise ValueError(
                f"model_v2_evaluation_report.json is missing {field_name}"
            )
        if not isinstance(evaluation_report[field_name], dict):
            raise ValueError(
                f"model_v2_evaluation_report.json {field_name} must be an object"
            )


def _load_joblib_member(artifact_archive: zipfile.ZipFile, member_name: str) -> Any:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / member_name
        with artifact_archive.open(member_name) as source_file:
            temp_path.write_bytes(source_file.read())
        try:
            return _load_joblib_path(temp_path)
        except Exception as exc:
            raise ValueError(
                f"Unable to load {member_name} with joblib: {exc}"
            ) from exc


def _load_joblib_path(path: Path) -> Any:
    if sys.platform != "win32":
        return joblib.load(path)

    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        return joblib.load(path)
    finally:
        pathlib.PosixPath = original_posix_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Model v2 CatBoost artifact zip reproducibility."
    )
    parser.add_argument(
        "--artifact-zip",
        required=True,
        help="Path to model_v2_catboost_artifacts.zip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_model_v2_catboost_artifacts(args.artifact_zip)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
