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

from dataset_streamlit_shell.ml.classification import (
    MODEL_KIND_LOGISTIC,
    MODEL_KIND_REGULARIZED,
    LogisticModelArtifact,
    RegularizedLogisticModelArtifact,
    artifact_from_payload,
    attach_logistic_test_costs,
    build_classification_agent_context,
    compute_cost_logistic,
    compute_cost_logistic_reg,
    confusion_matrix_counts,
    logistic_gradient_descent_steps,
    map_feature,
    predict_class_from_proba,
    predict_proba,
    predict_proba_from_logistic_artifact,
    predict_proba_from_regularized_artifact,
    sample_gradient_steps,
    save_classification_artifact,
    sigmoid,
    training_accuracy,
    validate_binary_target,
)
from dataset_streamlit_shell.ml.regression import apply_feature_scaler, create_feature_scaler

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


def test_university_admission_default_recipe_converges() -> None:
    """對齊 UI 教案：Z-score + α=0.001 + 10000 → Cost≈0.30、正確率明顯高於多數類基線。"""
    from dataset_streamlit_shell.ml.regression import (
        apply_standard_scaler,
        create_standard_scaler,
    )

    frame = pd.read_csv(ADMISSION_PATH)
    features = ["考試1分數", "考試2分數"]
    target = "是否錄取"
    scaler = create_standard_scaler(frame, features)
    feature_matrix = apply_standard_scaler(frame[features], scaler)
    steps = logistic_gradient_descent_steps(
        feature_matrix,
        frame[target],
        learning_rate=0.001,
        epochs=10000,
    )
    final = steps[-1]
    assert final.cost < 0.35
    assert final.cost < steps[0].cost
    proba = predict_proba(feature_matrix, final.weights, final.intercept)
    assert training_accuracy(frame[target], proba, 0.5) >= 85.0


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


def test_agent_context_reports_scaler_method_not_hardcoded_zscore() -> None:
    artifact = LogisticModelArtifact(
        model_kind=MODEL_KIND_LOGISTIC,
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        weights=[0.1, -0.2],
        intercept=0.0,
        scaler={"method": "minmax", "features": ["考試1分數", "考試2分數"]},
        training_cost=0.4,
        data_source="test",
    )
    context = build_classification_agent_context(
        page_name="邏輯迴歸",
        data_source="內建",
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        learning_rate=0.001,
        epochs=10000,
        row_count=100,
        artifact=artifact,
    )
    assert "minmax" in context
    assert "Z-score 特徵縮放" not in context


def test_regularized_gd_penalizes_weights_not_intercept() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]})
    kwargs = {
        "learning_rate": 0.1,
        "epochs": 1,
        "initial_weights": [1.0],
        "initial_intercept": 0.5,
    }
    plain = logistic_gradient_descent_steps(frame[["x"]], frame["y"], **kwargs)
    regularized = logistic_gradient_descent_steps(
        frame[["x"]],
        frame["y"],
        lambda_=2.0,
        regularized=True,
        **kwargs,
    )
    assert regularized[1].dj_db == pytest.approx(plain[1].dj_db)
    assert regularized[1].dj_dw is not None and plain[1].dj_dw is not None
    assert regularized[1].dj_dw[0] == pytest.approx(plain[1].dj_dw[0] + (2.0 / 2.0) * 1.0)


def test_logistic_gd_steps_include_micro_step_fields() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]})
    steps = logistic_gradient_descent_steps(
        frame[["x"]],
        frame["y"],
        learning_rate=0.1,
        epochs=2,
    )
    update = steps[1]
    assert update.prev_weights is not None
    assert update.prev_intercept is not None
    assert update.prev_cost is not None
    assert update.dj_dw is not None
    assert update.dj_db is not None
    assert update.delta_w is not None
    assert update.delta_b is not None
    assert len(update.dj_dw) == 1
    assert len(update.delta_w) == 1


def test_attach_logistic_test_costs_fills_each_step() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"x": [0.2, 0.8], "y": [0, 1]})
    steps = logistic_gradient_descent_steps(
        train[["x"]],
        train["y"],
        learning_rate=0.1,
        epochs=2,
    )
    annotated = attach_logistic_test_costs(steps, test[["x"]], test["y"])
    assert all(step.test_cost is not None for step in annotated)
    expected = compute_cost_logistic(test[["x"]], test["y"], steps[-1].weights, steps[-1].intercept)
    assert annotated[-1].test_cost == pytest.approx(expected)


def test_attach_logistic_test_costs_omits_lambda() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0], "y": [0, 1]})
    test = pd.DataFrame({"x": [0.2, 0.8], "y": [0, 1]})
    steps = logistic_gradient_descent_steps(
        train[["x"]],
        train["y"],
        learning_rate=0.1,
        epochs=2,
        lambda_=2.0,
        regularized=True,
    )
    annotated = attach_logistic_test_costs(steps, test[["x"]], test["y"])
    unreg = compute_cost_logistic(test[["x"]], test["y"], steps[-1].weights, steps[-1].intercept)
    regularized = compute_cost_logistic_reg(
        test[["x"]], test["y"], steps[-1].weights, steps[-1].intercept, 2.0
    )
    assert annotated[-1].test_cost == pytest.approx(unreg)
    assert annotated[-1].test_cost != pytest.approx(regularized)


def test_confusion_matrix_counts_is_actual_by_predicted() -> None:
    actual = pd.Series([0, 0, 1, 1, 1])
    predicted = pd.Series([0, 1, 0, 1, 1])
    assert confusion_matrix_counts(actual, predicted) == (1, 1, 1, 2)


def test_sample_gradient_steps_keeps_last_and_caps_length() -> None:
    from dataset_streamlit_shell.ml.regression import GradientDescentStep

    steps = [
        GradientDescentStep(
            iteration=i,
            weights=[0.0],
            intercept=0.0,
            cost=1.0,
            test_cost=1.1,
        )
        for i in range(10001)
    ]
    sampled = sample_gradient_steps(steps, max_points=80)
    assert 80 <= len(sampled) <= 82
    assert sampled[0] == steps[0]
    assert sampled[-1] == steps[-1]


def test_agent_context_includes_test_cost_curve_and_confusion() -> None:
    artifact = LogisticModelArtifact(
        model_kind=MODEL_KIND_LOGISTIC,
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        weights=[0.1, -0.2],
        intercept=0.0,
        scaler={"method": "minmax", "features": ["考試1分數", "考試2分數"]},
        training_cost=0.4,
        data_source="test",
    )
    context = build_classification_agent_context(
        page_name="邏輯迴歸",
        data_source="內建",
        features=["考試1分數", "考試2分數"],
        target="是否錄取",
        learning_rate=0.001,
        epochs=10000,
        row_count=100,
        artifact=artifact,
        lambda_=0.01,
        test_cost=0.55,
        cost_curve=[(0, 0.7, 0.72), (80, 0.4, 0.55)],
        confusion=(3, 1, 2, 4),
        train_accuracy=80.0,
        test_accuracy=70.0,
    )
    assert "最後訓練 Cost J：0.4" in context
    assert "最後測試 Cost J：0.55" in context
    assert "測試 Cost 為純對數損失，不含 λ" in context
    assert "0,0.7,0.72" in context
    assert "0→0=3" in context
    assert "訓練正確率：80%" in context
    assert "測試正確率：70%" in context


def test_regularized_artifact_scales_then_maps() -> None:
    base = ["檢測分數1", "檢測分數2"]
    frame = pd.DataFrame({base[0]: [0.4, 1.2], base[1]: [-0.2, 0.6]})
    scaler = create_feature_scaler(frame, base, "zscore")
    mapped_scaled, mapped_features = map_feature(
        apply_feature_scaler(frame, scaler),
        base,
        degree=6,
    )
    artifact = RegularizedLogisticModelArtifact(
        model_kind=MODEL_KIND_REGULARIZED,
        base_features=base,
        mapped_features=mapped_features,
        target="是否通過",
        weights=[0.01] * len(mapped_features),
        intercept=0.2,
        map_degree=6,
        lambda_=0.01,
        training_cost=0.5,
        data_source="test",
        scaler=scaler,
    )
    direct = predict_proba(mapped_scaled[mapped_features], artifact.weights, artifact.intercept)
    via_artifact = predict_proba_from_regularized_artifact(artifact, frame)
    assert list(via_artifact) == pytest.approx(list(direct))
