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

from dataset_streamlit_shell.ui.teaching_flow import (
    FLOW_INPUT,
    FLOW_MODEL,
    FLOW_OUTPUT,
    live_fit_caption,
    numeric_prediction_latex,
    regression_flow_svg,
    symbolic_prediction_latex,
    training_flow_state,
)


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
