from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ml.regression import GradientDescentStep, gradient_descent_steps
from dataset_streamlit_shell.ui.teaching_flow import (
    FLOW_INPUT,
    FLOW_MODEL,
    FLOW_OUTPUT,
    FLOW_VIEW_LABELS,
    FLOW_VIEW_MODEL,
    MICRO_COST,
    MICRO_GRAD,
    MICRO_PREDICT,
    MICRO_STEP_ORDER,
    MICRO_UPDATE,
    SAMPLE_OPS_HEAD,
    SAMPLE_OPS_SCALE_NOTE,
    build_training_micro_frames,
    gradient_board_rows,
    live_fit_caption,
    micro_stepper_html,
    numeric_prediction_latex,
    regression_flow_svg,
    sample_ops_cost_caption,
    sample_ops_table_rows,
    sample_ops_table_visible,
    sample_ops_x_labels,
    simple_gradient_board_lines,
    symbolic_prediction_latex,
    training_flow_state,
)
import pandas as pd


def test_symbolic_prediction_latex_simple_and_expanded_multiple() -> None:
    assert symbolic_prediction_latex(["城市人口_萬人"]) == r"Y = WX + B"
    assert (
        symbolic_prediction_latex(["面積_平方英尺", "房間數", "樓層數", "屋齡_年"])
        == r"Y = W_1 X_1 + W_2 X_2 + W_3 X_3 + W_4 X_4 + B"
    )


def test_numeric_prediction_latex_uses_concrete_weights() -> None:
    assert (
        numeric_prediction_latex(["x"], [1.25], 0.5)
        == r"Y = 1.25 X + 0.5"
    )
    assert (
        numeric_prediction_latex(["a", "b"], [2.0, -1.5], 3.0)
        == r"Y = 2 X_1 + -1.5 X_2 + 3"
    )


def test_live_fit_caption_reports_iteration_and_cost() -> None:
    caption = live_fit_caption(
        iteration=12,
        total_iterations=100,
        weights=[0.25],
        intercept=-1.5,
        cost=3.14159,
    )
    assert "12" in caption and "100" in caption
    assert "0.2500" in caption
    assert "-1.5000" in caption
    assert "3.1416" in caption


def test_training_flow_state_marks_model_hot_then_output_done() -> None:
    fitting = training_flow_state(finished=False)
    assert fitting.hot == FLOW_MODEL
    assert FLOW_INPUT in fitting.done
    assert FLOW_OUTPUT not in fitting.done

    done = training_flow_state(finished=True)
    assert done.hot is None
    assert {FLOW_INPUT, FLOW_MODEL, FLOW_OUTPUT} <= set(done.done)


def test_regression_flow_svg_includes_nodes_and_hot_class() -> None:
    svg = regression_flow_svg(hot=FLOW_MODEL, done={FLOW_INPUT}, live_caption="W=1")
    assert 'data-node="input"' in svg
    assert 'data-node="model"' in svg
    assert 'data-node="output"' in svg
    assert "輸入資料" in svg and "回歸模型" in svg and "輸出呈現" in svg
    assert 'class="node hot"' in svg or 'class="node hot ' in svg
    assert "W=1" in svg


def test_flow_view_labels_match_coarse_nodes_default_model() -> None:
    assert FLOW_VIEW_LABELS == ("輸入資料", "回歸模型", "輸出呈現")
    assert FLOW_VIEW_MODEL == "回歸模型"


def test_build_training_micro_frames_four_steps_per_update_iteration() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 5.0]})
    steps = gradient_descent_steps(
        frame[["x"]],
        frame["y"],
        learning_rate=0.1,
        epochs=2,
    )
    frames = build_training_micro_frames(steps, learning_rate=0.1)
    assert len(frames) == 2 * len(MICRO_STEP_ORDER)
    assert [f.micro_step for f in frames[:4]] == list(MICRO_STEP_ORDER)
    assert frames[0].iteration == 1
    assert frames[0].micro_step == MICRO_PREDICT
    assert frames[2].micro_step == MICRO_GRAD
    assert frames[2].dj_dw is not None
    assert frames[3].micro_step == MICRO_UPDATE
    assert frames[3].weights_after == steps[1].weights


def test_micro_stepper_html_marks_hot_step() -> None:
    html = micro_stepper_html(hot=MICRO_GRAD)
    assert "預測 ŷ" in html
    assert "算 Cost J" in html
    assert "算梯度" in html
    assert "更新參數" in html
    assert 'data-micro="grad"' in html
    assert "hot" in html


def test_simple_gradient_board_lines_reveal_by_micro_step() -> None:
    frame = _sample_micro_frame(micro=MICRO_PREDICT)
    predict_lines = simple_gradient_board_lines(frame)
    assert any("α" in line for line in predict_lines)
    assert any("w =" in line for line in predict_lines)
    assert not any("∂J/∂w" in line for line in predict_lines)

    grad_lines = simple_gradient_board_lines(_sample_micro_frame(micro=MICRO_GRAD))
    assert any("∂J/∂w" in line for line in grad_lines)
    assert not any("w'" in line for line in grad_lines)

    update_lines = simple_gradient_board_lines(_sample_micro_frame(micro=MICRO_UPDATE))
    assert any("Δw" in line for line in update_lines)
    assert any("w'" in line for line in update_lines)
    assert any("b'" in line for line in update_lines)


def test_gradient_board_rows_table_for_multiple_features() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import TrainingMicroFrame

    frame = TrainingMicroFrame(
        iteration=1,
        total_iterations=10,
        micro_step=MICRO_UPDATE,
        learning_rate=0.1,
        feature_names=["面積", "房間數"],
        weights_before=[1.0, 2.0],
        intercept_before=0.5,
        weights_after=[0.9, 1.8],
        intercept_after=0.4,
        cost_before=12.0,
        cost_after=10.0,
        dj_dw=[1.0, 2.0],
        dj_db=1.0,
        delta_w=[-0.1, -0.2],
        delta_b=-0.1,
    )
    rows = gradient_board_rows(frame)
    assert rows[0]["參數"] == "面積"
    assert rows[-1]["參數"] == "b（截距）"
    assert "梯度" in rows[0] and "更新後" in rows[-1]
    assert len(rows) == 3


def _sample_micro_frame(*, micro: str, feature_names: list[str] | None = None):
    from dataset_streamlit_shell.ui.teaching_flow import TrainingMicroFrame

    names = feature_names or ["x"]
    n = len(names)
    return TrainingMicroFrame(
        iteration=1,
        total_iterations=5,
        micro_step=micro,
        learning_rate=0.01,
        feature_names=names,
        weights_before=[0.5] * n,
        intercept_before=0.1,
        weights_after=[0.65] * n,
        intercept_after=0.4,
        cost_before=8.5,
        cost_after=7.0,
        dj_dw=[-6.5] * n,
        dj_db=-4.0,
        delta_w=[0.65] * n,
        delta_b=0.4,
    )


def test_sample_ops_table_visible_only_predict_and_cost() -> None:
    assert sample_ops_table_visible(MICRO_PREDICT)
    assert sample_ops_table_visible(MICRO_COST)
    assert not sample_ops_table_visible(MICRO_GRAD)
    assert not sample_ops_table_visible(MICRO_UPDATE)


def test_sample_ops_x_labels_mark_zscore() -> None:
    assert sample_ops_x_labels(["城市人口_萬人"]) == ["x（Z-score）"]
    assert sample_ops_x_labels(["面積", "房間數"]) == ["面積（Z-score）", "房間數（Z-score）"]
    assert "Z-score" in SAMPLE_OPS_SCALE_NOTE
    assert SAMPLE_OPS_HEAD == 5


def test_sample_ops_table_predict_columns_use_scaled_x() -> None:
    frame = _sample_micro_frame(micro=MICRO_PREDICT)
    # ŷ = 0.5 * (-1) + 0.1 = -0.4 ; ŷ = 0.5 * 1 + 0.1 = 0.6
    rows = sample_ops_table_rows(
        frame,
        scaled_x_rows=[[-1.0], [1.0]],
        y_rows=[0.0, 1.0],
    )
    assert rows is not None
    assert list(rows[0].keys()) == ["x（Z-score）", "ŷ", "y"]
    assert rows[0]["x（Z-score）"] == "-1"
    assert rows[0]["ŷ"] == "-0.4"
    assert rows[0]["y"] == "0"
    assert "error" not in rows[0]
    assert sample_ops_cost_caption(frame) is None


def test_sample_ops_table_cost_adds_error_and_batch_j() -> None:
    frame = _sample_micro_frame(micro=MICRO_COST)
    rows = sample_ops_table_rows(
        frame,
        scaled_x_rows=[[-1.0], [1.0]],
        y_rows=[0.0, 1.0],
    )
    assert rows is not None
    assert list(rows[0].keys()) == ["x（Z-score）", "ŷ", "y", "error", "error²"]
    # error = -0.4 - 0 = -0.4 ; error² = 0.16
    assert rows[0]["error"] == "-0.4"
    assert rows[0]["error²"] == "0.16"
    assert "J" not in rows[0]
    caption = sample_ops_cost_caption(frame)
    assert caption is not None
    assert "Cost J（整批）" in caption
    assert "8.5" in caption


def test_sample_ops_table_multiple_feature_columns() -> None:
    frame = _sample_micro_frame(micro=MICRO_PREDICT, feature_names=["面積", "房間數"])
    rows = sample_ops_table_rows(
        frame,
        scaled_x_rows=[[0.0, 1.0]],
        y_rows=[2.0],
    )
    assert rows is not None
    assert "面積（Z-score）" in rows[0]
    assert "房間數（Z-score）" in rows[0]
    # ŷ = 0.5*0 + 0.5*1 + 0.1 = 0.6
    assert rows[0]["ŷ"] == "0.6"


def test_sample_ops_table_hidden_returns_none() -> None:
    frame = _sample_micro_frame(micro=MICRO_GRAD)
    assert (
        sample_ops_table_rows(
            frame,
            scaled_x_rows=[[1.0]],
            y_rows=[1.0],
        )
        is None
    )


def test_classification_flow_labels_use_classifier_node() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import (
        CLASSIFICATION_FLOW_VIEW_LABELS,
        CLASSIFICATION_FLOW_VIEW_MODEL,
        classification_flow_svg,
    )

    assert CLASSIFICATION_FLOW_VIEW_LABELS == ("輸入資料", "分類模型", "輸出呈現")
    assert CLASSIFICATION_FLOW_VIEW_MODEL == "分類模型"
    svg = classification_flow_svg(hot=FLOW_MODEL, done={FLOW_INPUT})
    assert "分類模型" in svg
    assert "回歸模型" not in svg
    assert "輸入資料" in svg and "輸出呈現" in svg


def test_symbolic_logistic_prediction_splits_z_and_yhat() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import (
        symbolic_logistic_yhat_latex,
        symbolic_logistic_z_latex,
    )

    assert symbolic_logistic_z_latex(mapped=False) == r"z = w\cdot x+b"
    assert symbolic_logistic_z_latex(mapped=True) == r"z = w\cdot\phi(x)+b"
    yhat = symbolic_logistic_yhat_latex()
    assert r"\sigma(z)" in yhat
    assert r"\frac{1}{1+e^{-z}}" in yhat
    assert r"w\cdot x+b" not in yhat


def test_logistic_sample_ops_predict_uses_sigmoid() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import logistic_sample_ops_table_rows

    frame = _sample_micro_frame(micro=MICRO_PREDICT)
    # z = 0.5 * 0 + 0.1 = 0.1 ; ŷ = σ(0.1)
    rows = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=[[0.0]],
        y_rows=[1.0],
        show_x=True,
    )
    assert rows is not None
    assert list(rows[0].keys()) == ["x（Z-score）", "ŷ", "y"]
    expected = 1.0 / (1.0 + pow(2.718281828459045, -0.1))
    assert abs(float(rows[0]["ŷ"]) - expected) < 1e-6
    assert 0.0 < float(rows[0]["ŷ"]) < 1.0
    assert "error" not in rows[0]
    assert "ℓ" not in rows[0]


def test_logistic_sample_ops_cost_adds_ell_not_squared_error() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import logistic_sample_ops_table_rows

    frame = _sample_micro_frame(micro=MICRO_COST)
    rows = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=[[0.0]],
        y_rows=[1.0],
        show_x=True,
    )
    assert rows is not None
    assert "ℓ" in rows[0]
    assert "error" not in rows[0]
    assert "error²" not in rows[0]
    assert "J" not in rows[0]
    y_hat = float(rows[0]["ŷ"])
    ell = -1.0 * (1.0 * __import__("math").log(y_hat))
    assert abs(float(rows[0]["ℓ"]) - ell) < 1e-6


def test_logistic_sample_ops_yhat_changes_if_x_not_scaled() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import logistic_sample_ops_table_rows

    frame = _sample_micro_frame(micro=MICRO_PREDICT)
    scaled = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=[[-1.0]],
        y_rows=[1.0],
        show_x=True,
    )
    raw = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=[[50.0]],
        y_rows=[1.0],
        show_x=True,
    )
    assert scaled is not None and raw is not None
    assert scaled[0]["ŷ"] != raw[0]["ŷ"]


def test_logistic_sample_ops_poly_hides_mapped_x() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import logistic_sample_ops_table_rows

    frame = _sample_micro_frame(micro=MICRO_PREDICT, feature_names=["φ1", "φ2"])
    rows = logistic_sample_ops_table_rows(
        frame,
        model_x_rows=[[0.2, -0.1]],
        y_rows=[0.0],
        show_x=False,
    )
    assert rows is not None
    assert list(rows[0].keys()) == ["ŷ", "y"]
    assert "φ1" not in rows[0]


def test_regularized_compact_board_shows_lambda_and_reg_grad() -> None:
    from dataset_streamlit_shell.ui.teaching_flow import regularized_compact_board_lines

    frame = _sample_micro_frame(micro=MICRO_GRAD, feature_names=[f"φ{i}" for i in range(27)])
    lines = regularized_compact_board_lines(frame, lambda_=0.01)
    joined = "\n".join(lines)
    assert "λ = 0.01" in joined
    assert "‖w‖²" in joined
    assert "∂J/∂w" in joined
    assert "λ/m" in joined or r"\frac{\lambda}{m}" in joined
    assert "不加 λ" in joined
    assert joined.count("φ") < 5
