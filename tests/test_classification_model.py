from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.classification import (
    MODEL_KIND_LOGISTIC,
    MODEL_KIND_REGULARIZED,
    LogisticModelArtifact,
    RegularizedLogisticModelArtifact,
    artifact_from_payload,
    build_classification_agent_context,
    compute_cost_logistic,
    compute_cost_logistic_reg,
    logistic_gradient_descent_steps,
    map_feature,
    predict_class_from_proba,
    predict_proba,
    predict_proba_from_logistic_artifact,
    predict_proba_from_regularized_artifact,
    save_classification_artifact,
    sigmoid,
    training_accuracy,
    validate_binary_target,
)

ADMISSION_PATH = (
    TEMPLATE_ROOT
    / "dataset_streamlit_shell"
    / "built-in-data"
    / "classification"
    / "university_admission.csv"
)


def test_sigmoid_bounds() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(-500) < 0.01
    assert sigmoid(500) > 0.99


def test_map_feature_degree_six_has_27_columns() -> None:
    frame = pd.DataFrame({"x1": [0.5, 1.0], "x2": [1.5, 2.0]})
    mapped, columns = map_feature(frame, ["x1", "x2"], degree=6)
    assert mapped.shape == (2, 27)
    assert len(columns) == 27


def test_logistic_cost_decreases_with_gradient_descent() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0, 0, 1, 1]})
    steps = logistic_gradient_descent_steps(
        frame[["x"]],
        frame["y"],
        learning_rate=0.1,
        epochs=200,
    )
    assert steps[-1].cost < steps[0].cost


def test_regularized_cost_includes_lambda_term() -> None:
    frame = pd.DataFrame({"x1": [0.0, 1.0], "x2": [1.0, 0.0]})
    y = np.array([0.0, 1.0])
    w = np.array([1.0, 2.0])
    base = compute_cost_logistic(frame, y, w, 0.0)
    reg = compute_cost_logistic_reg(frame, y, w, 0.0, lambda_=1.0)
    assert reg > base


def test_logistic_artifact_json_round_trip(tmp_path: Path) -> None:
    artifact = LogisticModelArtifact(
        model_kind=MODEL_KIND_LOGISTIC,
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        weights=[0.5, 0.3],
        intercept=-1.0,
        scaler=None,
        training_cost=0.4,
        data_source="test",
    )
    path = tmp_path / "logistic.json"
    save_classification_artifact(artifact, path)
    loaded = artifact_from_payload(json.loads(path.read_text(encoding="utf-8")))
    frame = pd.DataFrame({"考試1分數": [50.0], "考試2分數": [60.0]})
    prob = float(predict_proba_from_logistic_artifact(loaded, frame).iloc[0])
    assert 0.0 <= prob <= 1.0


def test_regularized_artifact_inference(tmp_path: Path) -> None:
    base = ["檢測分數1", "檢測分數2"]
    frame = pd.DataFrame({base[0]: [0.1], base[1]: [0.7]})
    mapped, mapped_features = map_feature(frame, base, degree=6)
    artifact = RegularizedLogisticModelArtifact(
        model_kind=MODEL_KIND_REGULARIZED,
        base_features=base,
        mapped_features=mapped_features,
        target="是否通過",
        weights=[0.01] * len(mapped_features),
        intercept=0.0,
        map_degree=6,
        lambda_=0.01,
        training_cost=0.5,
        data_source="test",
    )
    path = tmp_path / "reg.json"
    save_classification_artifact(artifact, path)
    loaded = artifact_from_payload(
        json.loads(path.read_text(encoding="utf-8")),
        expected_kind=MODEL_KIND_REGULARIZED,
    )
    prob = float(predict_proba_from_regularized_artifact(loaded, frame).iloc[0])
    assert 0.0 <= prob <= 1.0


def test_artifact_kind_validation() -> None:
    payload = {
        "model_kind": MODEL_KIND_LOGISTIC,
        "features": ["a"],
        "target": "y",
        "weights": [0.0],
        "intercept": 0.0,
        "scaler": None,
        "training_cost": 1.0,
        "data_source": "t",
    }
    try:
        artifact_from_payload(payload, expected_kind=MODEL_KIND_REGULARIZED)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_threshold_changes_predicted_class() -> None:
    probability = pd.Series([0.4, 0.6])
    low = predict_class_from_proba(probability, 0.5).tolist()
    high = predict_class_from_proba(probability, 0.35).tolist()
    assert low == [0, 1]
    assert high == [1, 1]


def test_training_accuracy() -> None:
    actual = pd.Series([0, 1, 1, 0])
    probability = pd.Series([0.1, 0.9, 0.8, 0.2])
    assert training_accuracy(actual, probability, 0.5) == 100.0


def test_university_admission_csv_loads() -> None:
    frame = pd.read_csv(ADMISSION_PATH)
    assert len(frame) == 100
    assert validate_binary_target(frame["是否錄取"])


def test_build_classification_agent_context() -> None:
    context = build_classification_agent_context(
        page_name="邏輯迴歸",
        data_source="內建",
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        learning_rate=0.01,
        epochs=10,
        row_count=100,
        threshold=0.5,
    )
    assert "邏輯迴歸" in context
    assert "threshold" in context
