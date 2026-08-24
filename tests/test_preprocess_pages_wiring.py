from __future__ import annotations

from pathlib import Path

PAGES = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "pages"
)
APP = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "app.py"
)
UI = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "ui"
)


def test_new_preprocess_pages_exist() -> None:
    for name in (
        "17_Data_Transform.py",
        "20_Data_Split.py",
    ):
        assert (PAGES / name).is_file()
    assert not (PAGES / "1_Database.py").is_file()
    assert not (PAGES / "15_Data_Integration.py").is_file()


def test_app_nav_order_for_integration_transform_split() -> None:
    src = APP.read_text(encoding="utf-8")
    assert "1_Database.py" not in src
    assert "資料上傳與預覽" not in src
    assert "15_Data_Integration.py" not in src
    assert "8_Ready.py" not in src
    assert "建立 Ready 分析就緒資料" not in src
    collab = src.index('"AI 協作資料整理"')
    quality = src.index("3_Field_Quality.py")
    transform = src.index("17_Data_Transform.py")
    duplicates = src.index("4_Duplicates.py")
    scaling = src.index("13_Feature_Scaling.py")
    charts = src.index("2_Charts.py")
    split = src.index("20_Data_Split.py")
    assert collab < quality < transform < duplicates
    assert scaling < charts < split
    assert 'title="欄位與資料整合"' in src
    assert 'title="資料整合"' not in src


def test_guidance_strings_point_to_field_quality_not_upload() -> None:
    data_ui = (UI / "data_ui.py").read_text(encoding="utf-8")
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    charts = (PAGES / "2_Charts.py").read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    for src in (data_ui, workflow, charts, app):
        assert "資料上傳與預覽" not in src
    assert "欄位與資料整合" in data_ui
    assert "欄位與資料整合" in workflow
    assert "欄位與資料整合" in app
    assert "欄位與資料概覽" not in workflow
    assert "建立 Ready 分析就緒資料" not in charts
    assert "欄位與資料整合" in charts


def test_quality_page_supports_dual_and_clear() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "working_dataset_file_exists" in src
    assert "clear_to_dual_start" in src
    assert "_render_dual_table_quality" in src
    assert "render_integration_page" not in src
    assert "integration_apply" not in src
    assert "套用合併並寫入工作資料" not in src
    assert "重新讀取雙表" in src
    assert "恢復內建雙表" in src
    assert "訓練前預測（資料整合）" in src
    assert "render_transform_page" in src
    assert "render_split_page" in src
    assert "transform_column_overview" in src
    assert "flagged_synonym_columns" in src
    assert "欄位輔助資訊" in src
    assert "同義提示" in src
    assert "請選擇要關注的文字欄" in src
    assert "clear_split_datasets" in (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "save_split_datasets" in (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "clear_dual_table_copies" in (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "ensure_dual_table_copies" in src


def test_charts_page_reads_working_only() -> None:
    src = (PAGES / "2_Charts.py").read_text(encoding="utf-8")
    assert "load_working_dataset" in src
    assert "load_ready_dataset" not in src
    assert "chart_data_source" not in src
    assert "Working 工作資料" in src
    assert "Original 原始資料" not in src
    assert "本頁只讀 Ready" not in src
    assert "本頁只讀 Working" in src


def test_split_page_reads_working() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "render_split_page" in src
    split = src[src.index("def render_split_page") :]
    assert "load_working_dataset" in split
    assert "load_ready_dataset" not in split
    assert "Working 工作資料" in split or "工作資料" in split
    assert "尚未建立 Ready" not in split


def test_pca_shell_reads_working() -> None:
    pca = (PAGES / "10_PCA.py").read_text(encoding="utf-8")
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    shell = workflow[workflow.index("def render_analysis_shell") :]
    next_def = shell.find("\ndef ", 1)
    shell = shell[: next_def if next_def != -1 else len(shell)]
    assert "load_working_dataset" in shell
    assert "load_ready_dataset" not in shell
    assert "Ready 分析就緒資料" not in pca
    assert "Working 工作資料" in pca


def test_ready_page_is_removed() -> None:
    assert not (PAGES / "8_Ready.py").exists()
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "def render_ready_page" not in workflow
    assert "create_ready_dataset" not in workflow


def test_overview_downloads_working_not_ready() -> None:
    src = APP.read_text(encoding="utf-8")
    assert "下載 Working" in src
    assert "load_ready_dataset" not in src
    assert "READY_DATASET_PATH" not in src
    assert "建立 Ready 分析就緒資料" not in src
