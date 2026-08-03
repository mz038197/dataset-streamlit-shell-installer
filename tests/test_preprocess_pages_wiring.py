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


def test_app_nav_order_for_integration_transform_split() -> None:
    src = APP.read_text(encoding="utf-8")
    quality = src.index("3_Field_Quality.py")
    integration = src.index("15_Data_Integration.py")
    transform = src.index("17_Data_Transform.py")
    duplicates = src.index("4_Duplicates.py")
    ready = src.index("8_Ready.py")
    charts = src.index("2_Charts.py")
    split = src.index("20_Data_Split.py")
    assert quality < integration < transform < duplicates
    assert ready < charts < split


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
