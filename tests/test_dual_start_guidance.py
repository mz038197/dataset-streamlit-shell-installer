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
APP = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "app.py"
)


def test_empty_snapshot_guidance_points_to_field_quality() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert '請先提醒到「欄位與資料整合」' in src
    assert "請先提醒到「資料上傳與預覽」上傳" not in src
    assert "再到「資料整合」合併後再整理" not in src


def test_base_context_describes_merge_origin_not_upload() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "上傳資料時系統會先建立" not in src
    assert "資料整合」套用合併時一併建立" not in src
    assert "資料整合成功時與 Original 一併寫入" in src
    assert "workspace/integration" in src
    assert "重新讀取工作資料" in src
    assert "不要另存新檔" in src


def test_chat_panel_has_no_empty_workspace_info_banner() -> None:
    """資料 Agent 欄不顯示「尚未建立工作資料」橫幅；導引留在主教學欄。"""
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "尚未上傳 CSV" not in src
    assert "你仍可啟用 Agent 詢問一般問題" not in src
    panel = src.split("def render_chat_panel", 1)[1].split("\ndef ", 1)[0]
    assert "尚未建立工作資料" not in panel


def test_overview_mentions_dual_start_not_upload_page() -> None:
    src = APP.read_text(encoding="utf-8")
    assert "資料上傳與預覽" not in src
    assert "欄位與資料整合" in src
    assert "欄位與資料概覽" not in src
    assert "雙表起點" in src
    assert "15_Data_Integration.py" not in src


def test_dual_start_page_has_no_missing_working_info_banner() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "目前還沒有 Working 工作資料" not in src
    assert "以下並排顯示雙表工作副本" not in src
    assert "答對後請 Agent 合併寫入 Original 與 Working" in src
    assert "請 Agent 對齊鍵名後合併" in src
