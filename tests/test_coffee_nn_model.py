from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.coffee_nn import (
    BUILTIN_DATA_PATH_SUFFIX,
    CompileSpec,
    HiddenLayerSpec,
    LOSS_AUTO,
    NetworkSpec,
    TrainConfig,
    build_sequential_model,
    estimate_parameter_count,
    format_model_code,
    lab02_default_compile_spec,
    lab02_default_network_spec,
    load_builtin_frame,
    predict_class_labels,
    predict_scores,
    train_model,
    validate_network_spec,
)

BUILTIN_PATH = TEMPLATE_ROOT / "dataset_streamlit_shell" / Path(*BUILTIN_DATA_PATH_SUFFIX)


def test_builtin_data_has_400_rows_and_binary_labels() -> None:
    frame = load_builtin_frame(BUILTIN_PATH)
    assert len(frame) == 400
    labels = frame["類別"].astype(int).unique()
    assert set(labels.tolist()) == {0, 1}


def test_validate_rejects_too_many_output_units() -> None:
    frame = load_builtin_frame(BUILTIN_PATH)
    spec = NetworkSpec(
        input_features=("特徵1", "特徵2"),
        hidden_layers=(),
        output_units=3,
        output_activation="softmax",
    )
    errors = validate_network_spec(spec, frame)
    assert any("不相容" in message for message in errors)


def test_validate_manual_loss_binary_requires_sigmoid() -> None:
    frame = load_builtin_frame(BUILTIN_PATH)
    spec = NetworkSpec(
        input_features=("特徵1", "特徵2"),
        hidden_layers=(),
        output_units=1,
        output_activation="relu",
        loss_choice="BinaryCrossentropy",
    )
    errors = validate_network_spec(spec, frame)
    assert any("sigmoid" in message for message in errors)


def test_format_model_code_contains_sequential() -> None:
    spec = lab02_default_network_spec()
    compile_spec = lab02_default_compile_spec()
    code = format_model_code(spec, compile_spec)
    assert "Sequential" in code
    assert "Dense(3, activation='sigmoid')" in code


def test_build_and_estimate_parameters() -> None:
    pytest.importorskip("tensorflow")
    spec = lab02_default_network_spec()
    assert estimate_parameter_count(spec) == build_sequential_model(spec).count_params()


def test_train_model_smoke() -> None:
    pytest.importorskip("tensorflow")
    frame = load_builtin_frame(BUILTIN_PATH)
    spec = NetworkSpec(
        input_features=("特徵1", "特徵2"),
        hidden_layers=(HiddenLayerSpec(2, "relu"),),
        output_units=1,
        output_activation="sigmoid",
        loss_choice=LOSS_AUTO,
    )
    compile_spec = CompileSpec(optimizer_name="Adam", learning_rate=0.05)
    train_config = TrainConfig(epochs=2, tile_factor=10, random_seed=0)
    x = frame[["特徵1", "特徵2"]].to_numpy(dtype=np.float32)
    y = frame["類別"].to_numpy(dtype=np.float32)
    artifacts = train_model(spec, compile_spec, train_config, x, y)
    assert artifacts.result.parameter_count > 0
    assert len(artifacts.result.history.get("loss", [])) == 2
    scores = predict_scores(
        artifacts.model,
        x[:5],
        spec,
        feature_normalizer=artifacts.feature_normalizer,
    )
    preds = predict_class_labels(scores, spec)
    assert preds.shape == (5,)
