from __future__ import annotations

from pathlib import Path

UI_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "ui"
    / "logistic_regression_ui.py"
)


def test_logistic_pages_omit_save_manual_and_page_intro_info() -> None:
    src = UI_PATH.read_text(encoding="utf-8")
    assert "保存模型 JSON" not in src
    assert "手動預測" not in src
    assert "依兩科考試成績預測是否錄取" not in src
    assert "晶片兩項檢測分數預測是否通過" not in src
