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
        "15_Data_Integration.py",
        "17_Data_Transform.py",
        "20_Data_Split.py",
    ):
        assert (PAGES / name).is_file()
    assert not (PAGES / "1_Database.py").is_file()


def test_app_nav_order_for_integration_transform_split() -> None:
    src = APP.read_text(encoding="utf-8")
    assert "1_Database.py" not in src
    assert "資料上傳與預覽" not in src
    collab = src.index('"AI 協作資料整理"')
    quality = src.index("3_Field_Quality.py")
    integration = src.index("15_Data_Integration.py")
    transform = src.index("17_Data_Transform.py")
    duplicates = src.index("4_Duplicates.py")
    ready = src.index("8_Ready.py")
    charts = src.index("2_Charts.py")
    split = src.index("20_Data_Split.py")
    assert collab < quality < integration < transform < duplicates
    assert ready < charts < split


def test_guidance_strings_point_to_field_quality_not_upload() -> None:
    data_ui = (UI / "data_ui.py").read_text(encoding="utf-8")
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    charts = (PAGES / "2_Charts.py").read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    for src in (data_ui, workflow, charts, app):
        assert "資料上傳與預覽" not in src
    assert "欄位與資料概覽" in data_ui
    assert "欄位與資料概覽" in workflow
    assert "欄位與資料概覽" in app
    assert "建立 Ready 分析就緒資料" in charts


def test_quality_page_supports_dual_and_clear() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "working_dataset_file_exists" in src
    assert "clear_to_dual_start" in src
    assert "_render_dual_table_quality" in src
    assert "render_integration_page" in src
    assert "render_transform_page" in src
    assert "render_split_page" in src
    assert "clear_split_datasets" in (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "save_split_datasets" in (UI / "data_ui.py").read_text(encoding="utf-8")


def test_charts_page_reads_ready_only() -> None:
    src = (PAGES / "2_Charts.py").read_text(encoding="utf-8")
    assert "load_ready_dataset" in src
    assert "chart_data_source" not in src
    assert "Working 工作資料" not in src
    assert "Original 原始資料" not in src
    assert "本頁只讀 Ready" in src
