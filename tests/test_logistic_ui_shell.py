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


def test_logistic_model_formula_splits_z_and_sigmoid() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    assert "symbolic_logistic_z_latex" in ui_src
    assert "symbolic_logistic_yhat_latex" in ui_src
    assert r"\mathrm{sigmoid}" not in ui_src
    assert "MODEL_FORMULA_LATEX" not in ui_src


def test_logistic_ui_uses_decision_slots_not_teaching_flow() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    assert "決策槽" in ui_src
    assert "slot_button_label" in ui_src
    assert "st.columns(6)" in ui_src
    assert "模型程式碼預覽" in ui_src
    assert "after_reply=" in ui_src
    assert "consume_train_request" in ui_src
    assert "測試集混淆矩陣" in ui_src
    assert "測試集機率對照" not in ui_src
    assert "precision_recall_f1_caption" in ui_src
    assert "predicted_class" in ui_src
    assert "ŷ=0.5" in ui_src
    assert "Sigmoid 函數視覺化" in ui_src
    assert "classification_flow_svg" not in ui_src
    assert "TEACHING_FLOW_CSS" not in ui_src
    assert "目前查看" not in ui_src
    assert "逐步模式" not in ui_src
    assert "梯度演算板" not in ui_src
    assert "樣本運算表" not in ui_src
    assert "micro_stepper_html" not in ui_src
    assert "render_dataset_metrics" not in ui_src
    assert "number_input" not in ui_src


def test_logistic_cost_formula_still_available() -> None:
    ui_src = UI_PATH.read_text(encoding="utf-8")
    assert "COST_J_LOGISTIC_LATEX" in ui_src
    assert "COST_GD_W_LOGISTIC_LATEX" in ui_src or "COST_GD_W_LATEX" in ui_src
    assert "導數項" in ui_src
    assert "梯度下降演算法" in ui_src


def test_training_animation_replays_frames_in_one_request() -> None:
    src = UI_PATH.read_text(encoding="utf-8")
    marker = "def _run_training("
    start = src.index(marker)
    body = []
    for line in src[start:].splitlines()[1:]:
        if line.startswith("def "):
            break
        body.append(line)
    text = "\n".join(body)
    assert "for step in sampled:" in text
    assert "st.rerun()" not in text
