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
    compute_hinge_loss,
    decision_function_from_artifact,
    fit_linear_svc,
    linear_svm_gradient_descent_steps,
    predict_binary_class,
    predict_class_from_artifact,
    save_svm_artifact,
    support_vector_candidates,
    validate_svm_target,
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


@pytest.fixture
def signed_blobs_frame(blobs_frame: pd.DataFrame) -> pd.DataFrame:
    frame = blobs_frame.copy()
    frame["類別"] = np.where(frame["類別"].to_numpy(dtype=int) == 1, 1, -1)
    return frame


def test_fit_linear_svc_has_support_vectors(signed_blobs_frame: pd.DataFrame) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = signed_blobs_frame[features]
    y = signed_blobs_frame[target]
    clf = fit_linear_svc(x, y, C=1.0)
    assert clf.support_vectors_.shape[0] > 0


def test_artifact_json_round_trip(signed_blobs_frame: pd.DataFrame, tmp_path: Path) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = signed_blobs_frame[features]
    y = signed_blobs_frame[target]
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
    scores = decision_function_from_artifact(loaded, signed_blobs_frame.head(5))
    assert scores.shape == (5,)
    predicted = predict_class_from_artifact(loaded, signed_blobs_frame.head(5))
    assert set(predicted.tolist()).issubset({-1, 1})


def test_artifact_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError, match="expected model_kind"):
        artifact_from_payload({"model_kind": "logistic_regression"}, expected_kind=MODEL_KIND_LINEAR_SVM)


def test_predictions_match_fitted_svc(signed_blobs_frame: pd.DataFrame) -> None:
    features = ["特徵1", "特徵2"]
    target = "類別"
    x = signed_blobs_frame[features].to_numpy(dtype=float)
    y = signed_blobs_frame[target].to_numpy(dtype=float)
    clf = fit_linear_svc(signed_blobs_frame[features], signed_blobs_frame[target], C=1.0)
    artifact = build_linear_svm_artifact(
        clf,
        features=features,
        target=target,
        C=1.0,
        scaler=None,
        data_source="test",
        feature_frame=signed_blobs_frame[features],
        target_series=signed_blobs_frame[target],
    )
    svc_pred = clf.predict(x)
    artifact_pred = predict_class_from_artifact(artifact, signed_blobs_frame)
    assert np.mean(svc_pred == np.asarray(artifact_pred)) == 1.0


def test_validate_svm_target_accepts_only_negative_one_and_positive_one() -> None:
    assert validate_svm_target(pd.Series([-1, 1, -1, 1]))
    assert not validate_svm_target(pd.Series([0, 1, 0, 1]))


def test_predict_binary_class_returns_signed_labels() -> None:
    predicted = predict_binary_class(np.array([-0.5, 0.0, 2.0]))
    assert predicted.tolist() == [-1, 1, 1]


def test_hinge_loss_training_steps_reduce_loss() -> None:
    frame = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 2.0, 6.0, 7.0, 8.0],
            "x2": [2.0, 3.0, 1.0, 5.0, 7.0, 6.0],
            "y": [-1, -1, -1, 1, 1, 1],
        }
    )
    steps = linear_svm_gradient_descent_steps(
        frame[["x1", "x2"]],
        frame["y"],
        learning_rate=0.001,
        C=1.0,
        epochs=200,
    )
    assert steps[-1].cost < steps[0].cost
    supports = support_vector_candidates(frame[["x1", "x2"]], frame["y"], steps[-1].weights, steps[-1].intercept)
    assert supports.any()


def test_compute_hinge_loss_matches_margin_formula() -> None:
    x = pd.DataFrame({"x1": [1.0, 3.0], "x2": [1.0, 3.0]})
    y = pd.Series([-1, 1])
    loss = compute_hinge_loss(x, y, [1.0, 1.0], 0.0, C=1.0)
    assert loss == pytest.approx(2.5)
