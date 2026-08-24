from __future__ import annotations

import re
from pathlib import Path

UI = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "ui"
)


def test_linear_regression_page_uses_decision_slots_not_old_skeleton() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    assert "決策槽" in src
    assert "slot_button_label" in src
    assert "模型程式碼預覽" in src
    assert "after_reply=" in src
    assert "consume_train_request" in src
    assert "套用預設" not in src
    assert "目前查看" not in src
    assert "逐步模式" not in src
    assert "梯度演算板" not in src
    assert "樣本運算表" not in src
    assert "regression_flow_svg" not in src
    assert "micro_stepper_html" not in src
    assert "保存模型 JSON" not in src
    assert "手動預測" not in src
    assert "檢視中" not in src
    assert "render_dataset_metrics" not in src
    assert "DATA_PREVIEW_ROWS" in src
    assert "st.columns(6)" in src
    assert "split_frame_by_train_pct" in src
    assert "test_cost" in src


def test_workflow_ui_no_longer_owns_linear_regression_page() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "def render_linear_regression_page" not in src
    assert "逐步模式" not in src
    assert "梯度演算板" not in src


def test_logistic_page_still_uses_teaching_flow() -> None:
    src = (UI / "logistic_regression_ui.py").read_text(encoding="utf-8")
    assert "目前查看" in src
    assert "逐步模式" in src
    assert "梯度演算板" in src
    assert "樣本運算表" in src
    assert "decision slot" not in src.lower()
    assert "決策槽" not in src


def test_dataset_base_context_does_not_embed_lr_host_fragment() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "def teaching_page_host_context" in src
    assert "lr_host_context_fragment" not in src
    assert "lr_slots.json" not in src
    assert "lr_train_request.json" not in src


def _def_block(src: str, name: str) -> str:
    marker = f"def {name}("
    start = src.index(marker)
    lines = src[start:].splitlines()
    collected = [lines[0]]
    for line in lines[1:]:
        if line.startswith("def "):
            break
        collected.append(line)
    return "\n".join(collected)


def test_training_animation_replays_frames_in_one_request() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    simple = _def_block(src, "_run_simple_training")
    multiple = _def_block(src, "_run_multiple_training")
    maybe_start = _def_block(src, "_maybe_start_from_request")
    for body in (simple, multiple):
        assert "for step in sampled:" in body
        assert "st.rerun()" not in body
        assert 'anim["index"] = index + 1' not in body
    assert "st.rerun()" not in maybe_start
    assert "if train_clicked and allowed:\n        _start()\n        st.rerun()" not in src


def test_training_charts_share_near_square_canvas() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    match = re.search(
        r"LR_TRAINING_CHART_FIGSIZE = \(([\d.]+),\s*([\d.]+)\)",
        src,
    )
    assert match is not None
    width, height = float(match.group(1)), float(match.group(2))
    assert width == height
    assert width >= 6
    simple = _def_block(src, "_render_simple_step_plot")
    cost = _def_block(src, "_render_cost_history_plot")
    actual = _def_block(src, "_render_test_prediction_plot")
    left = _def_block(src, "_render_left_prediction_plot")
    for body in (simple, cost, actual, left):
        assert "figsize=LR_TRAINING_CHART_FIGSIZE" in body
        assert "figsize=(8, 4.8)" not in body
        assert "figsize=(6.6, 5.2)" not in body
    assert 'set_aspect("equal")' in actual or "set_aspect('equal')" in actual
    assert 'set_aspect("equal")' in left or "set_aspect('equal')" in left


def test_test_prediction_plot_locks_limits_when_predictions_explode() -> None:
    import sys

    import pandas as pd

    template = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "add_dataset_streamlit_shell"
        / "templates"
    )
    if str(template) not in sys.path:
        sys.path.insert(0, str(template))
    from dataset_streamlit_shell.ui.lr_ui import _render_test_prediction_plot

    class _Hold:
        fig = None

        def pyplot(self, fig, clear_figure=True):
            self.fig = fig

    holder = _Hold()
    _render_test_prediction_plot(
        pd.Series([2.0, 8.0]),
        pd.Series([0.0, 999.0]),
        "獲利",
        holder,
    )
    ax = holder.fig.axes[0]
    assert ax.get_title() == "測試集預測對照"
    assert ax.get_xlabel() == "實際 獲利"
    assert ax.get_ylabel() == "預測 獲利"
    assert ax.get_xlim() == (0.0, 8.0)
    assert ax.get_ylim() == (0.0, 8.0)


def test_training_layout_adds_test_prediction_plot_below() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    assert "測試集預測對照" in src
    assert "訓練畫面只有回歸線與訓練／測試 Cost。" not in src
    simple_stage = _def_block(src, "_render_simple_stage")
    multiple_stage = _def_block(src, "_render_multiple_stage")
    for body in (simple_stage, multiple_stage):
        assert "chart_left, chart_right = st.columns(2)" in body
        assert "pred_placeholder = st.empty()" in body
        assert "訓練後這裡只會出現回歸線與訓練／測試 Cost。" not in body
        assert "測試集預測對照" in body
    simple_train = _def_block(src, "_run_simple_training")
    multiple_train = _def_block(src, "_run_multiple_training")
    for body in (simple_train, multiple_train):
        assert "_render_step_test_prediction(" in body
        assert "_render_cost_history_plot(" in body


def test_left_scatter_marks_held_out_points() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    simple = _def_block(src, "_render_simple_step_plot")
    multiple = _def_block(src, "_render_left_prediction_plot")
    for body in (simple, multiple):
        assert 'label="訓練點"' in body
        assert 'label="測試點"' in body
    assert "alpha=0.4" in simple or "alpha=0.40" in simple
    test_plot = _def_block(src, "_render_test_prediction_plot")
    assert 'label="測試點"' not in test_plot
    assert 'set_title("測試集預測對照")' in test_plot
    assert 'set_xlabel(f"實際 {target}")' in test_plot
    assert 'set_ylabel(f"預測 {target}")' in test_plot
    assert "prediction_axis_limits" in test_plot
    assert "min(actual.min(), prediction.min())" not in test_plot
    assert 'label="完全預測正確"' in test_plot
    assert "figsize=LR_TRAINING_CHART_FIGSIZE" in test_plot
