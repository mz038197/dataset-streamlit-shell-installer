from __future__ import annotations

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


def test_dataset_base_context_includes_lr_host_fragment() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "lr_host_context_fragment" in src
    assert "lr_slots.json" in src
    assert "lr_train_request.json" in src
