from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.svm import (
    MODEL_KIND_LINEAR_SVM,
    artifact_from_payload,
    build_linear_svm_artifact,
    decision_function_from_artifact,
    fit_linear_svc,
    predict_class_from_artifact,
    save_svm_artifact,
)

BLOBS_CSV = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "svm_blobs_80.csv"
)


@pytest.fixture
def blobs_frame() -> pd.DataFrame:
    return pd.read_csv(BLOBS_CSV)


def test_fit_linear_svc_has_support_vectors(blobs_frame: pd.DataFrame) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = blobs_frame[features]
    y = blobs_frame[target]
    clf = fit_linear_svc(x, y, C=1.0)
    assert clf.support_vectors_.shape[0] > 0


def test_artifact_json_round_trip(blobs_frame: pd.DataFrame, tmp_path: Path) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = blobs_frame[features]
    y = blobs_frame[target]
    clf = fit_linear_svc(x, y, C=1.0)
    artifact = build_linear_svm_artifact(
        clf,
        features=features,
        target=target,
        C=1.0,
        scaler=None,
        data_source="test",
        feature_frame=x,
        target_series=y,
    )
    path = tmp_path / "model.json"
    save_svm_artifact(artifact, path)
    loaded = artifact_from_payload(json.loads(path.read_text(encoding="utf-8")))
    assert loaded.model_kind == MODEL_KIND_LINEAR_SVM
    assert loaded.n_support == artifact.n_support
    scores = decision_function_from_artifact(loaded, blobs_frame.head(5))
    assert scores.shape == (5,)
    predicted = predict_class_from_artifact(loaded, blobs_frame.head(5))
    assert set(predicted.tolist()).issubset({0, 1})


def test_artifact_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="expected model_kind"):
        artifact_from_payload({"model_kind": "logistic_regression"}, expected_kind=MODEL_KIND_LINEAR_SVM)


def test_predictions_match_fitted_svc(blobs_frame: pd.DataFrame) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = blobs_frame[features].to_numpy(dtype=float)
    y = blobs_frame[target].to_numpy(dtype=float)
    clf = fit_linear_svc(blobs_frame[features], blobs_frame[target], C=1.0)
    artifact = build_linear_svm_artifact(
        clf,
        features=features,
        target=target,
        C=1.0,
        scaler=None,
        data_source="test",
        feature_frame=blobs_frame[features],
        target_series=blobs_frame[target],
    )
    svc_pred = clf.predict(x)
    artifact_pred = predict_class_from_artifact(artifact, blobs_frame)
    assert np.mean(svc_pred == np.asarray(artifact_pred)) == 1.0
