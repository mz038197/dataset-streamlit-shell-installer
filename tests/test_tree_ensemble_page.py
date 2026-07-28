from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "add_dataset_streamlit_shell" / "templates" / "dataset_streamlit_shell"


def test_app_sidebar_merges_tree_ensemble_page() -> None:
    app_src = (TEMPLATE / "app.py").read_text(encoding="utf-8")
    assert 'title="決策樹與集成"' in app_src
    assert 'title="決策樹概念"' not in app_src
    assert 'title="XGBoost"' not in app_src
    assert "20_XGBoost.py" not in app_src
    assert not (TEMPLATE / "pages" / "20_XGBoost.py").exists()
    assert not (TEMPLATE / "ui" / "xgboost_ui.py").exists()


def test_tree_ui_has_three_learning_stages() -> None:
    ui_src = (TEMPLATE / "ui" / "tree_ui.py").read_text(encoding="utf-8")
    assert "單顆決策樹" in ui_src
    assert "隨機森林（Bagging）" in ui_src
    assert "XGBoost（Boosting）" in ui_src
    assert "horizontal=True" in ui_src
    assert "bagging_vs_boosting_contrast_markdown" in ui_src
    assert "learning_rate" in ui_src
    assert "forest_val_accuracy" in ui_src or "forest_validation_baseline" in ui_src
    assert "開始訓練" in ui_src
    assert 'st.tabs' not in ui_src
    assert "決策樹與集成" in ui_src
