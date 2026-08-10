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
    build_training_micro_frames,
    gradient_board_rows,
    live_fit_caption,
    micro_stepper_html,
    numeric_prediction_latex,
    regression_flow_svg,
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


def _sample_micro_frame(*, micro: str):
    from dataset_streamlit_shell.ui.teaching_flow import TrainingMicroFrame

    return TrainingMicroFrame(
        iteration=1,
        total_iterations=5,
        micro_step=micro,
        learning_rate=0.01,
        feature_names=["x"],
        weights_before=[0.0],
        intercept_before=0.0,
        weights_after=[0.65],
        intercept_after=0.4,
        cost_before=8.5,
        cost_after=7.0,
        dj_dw=[-6.5],
        dj_db=-4.0,
        delta_w=[0.65],
        delta_b=0.4,
    )
