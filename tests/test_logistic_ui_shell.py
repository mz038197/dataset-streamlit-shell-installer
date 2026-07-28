from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "add_dataset_streamlit_shell" / "templates" / "dataset_streamlit_shell"
UI_PATH = TEMPLATE / "ui" / "logistic_regression_ui.py"


def test_logistic_pages_omit_save_manual_and_page_intro_info() -> None:
    src = UI_PATH.read_text(encoding="utf-8")
    assert "保存模型 JSON" not in src
    assert "手動預測" not in src
    assert "依兩科考試成績預測是否錄取" not in src
    assert "晶片兩項檢測分數預測是否通過" not in src


def test_app_sidebar_merges_regularized_into_logistic() -> None:
    app_src = (TEMPLATE / "app.py").read_text(encoding="utf-8")
    assert 'title="邏輯迴歸"' in app_src
    assert 'title="正則化邏輯迴歸"' not in app_src
    assert "17_Regularized_Logistic_Regression.py" not in app_src
    assert not (TEMPLATE / "pages" / "17_Regularized_Logistic_Regression.py").exists()


def test_logistic_ui_has_two_learning_stages() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    quiz_src = (TEMPLATE / "ui" / "logistic_quiz.py").read_text(encoding="utf-8")
    assert "線性邊界" in quiz_src
    assert "多項式與 λ" in quiz_src
    assert "LEARNING_STAGES" in ui_src
    assert "horizontal=True" in ui_src
    assert "學習階段" in ui_src
    assert "st.tabs" not in ui_src


def test_logistic_model_formula_uses_explicit_sigmoid_fraction() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    assert r"\mathrm{sigmoid}" not in ui_src
    assert r"\frac{1}{1+e^{-(w\cdot x+b)}}" in ui_src or r"\frac{1}{1+e^{-(w \cdot x + b)}}" in ui_src


def test_logistic_cost_expander_matches_linear_regression_layout() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    assert '成本與梯度下降' in ui_src
    assert "COST_J_LOGISTIC_LATEX" in ui_src
    assert "COST_GD_W_LATEX" in ui_src or "COST_GD_W_LOGISTIC_LATEX" in ui_src
    assert "導數項" in ui_src
    assert "梯度下降演算法" in ui_src
