from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui import data_science  # noqa: E402

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
    / "ui"
    / "overview_ui.py"
)


def test_empty_snapshot_guidance_points_to_field_quality(tmp_path, monkeypatch) -> None:
    class _Session(dict):
        def pop(self, key, default=None):
            return dict.pop(self, key, default)

    monkeypatch.setattr(data_science.st, "session_state", _Session())
    monkeypatch.setattr(data_science, "WORKING_DATASET_PATH", tmp_path / "working.csv")
    monkeypatch.setattr(data_science, "ORIGINAL_DATASET_PATH", tmp_path / "original.csv")
    monkeypatch.setattr(data_science, "READY_DATASET_PATH", tmp_path / "ready.csv")
    monkeypatch.setattr(data_science, "SPLIT_DATASET_PATHS", ())
    text = data_science.working_page_snapshot()
    assert '請先提醒到「欄位與資料整合」' in text
    assert "請先提醒到「資料上傳與預覽」上傳" not in text
    assert "再到「資料整合」合併後再整理" not in text


def test_base_context_describes_merge_origin_not_upload() -> None:
    text = data_science.data_science_pack_layer()
    assert "上傳資料時系統會先建立" not in text
    assert "資料整合」套用合併時一併建立" not in text
    assert "資料整合成功時與 Original 一併寫入" in text
    assert "workspace/integration" in text
    assert "重新讀取工作資料" in text
    assert "不要另存新檔" in text


def test_chat_panel_has_no_empty_workspace_info_banner() -> None:
    """資料 Agent 欄不顯示「尚未建立工作資料」橫幅；導引留在主教學欄。"""
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "尚未上傳 CSV" not in src
    assert "你仍可啟用 Agent 詢問一般問題" not in src
    panel = src.split("def render_chat_panel", 1)[1].split("\ndef ", 1)[0]
    assert "尚未建立工作資料" not in panel


def test_overview_introduces_framework_only() -> None:
    """總覽只介紹 AI Lab 框架，不提 Data Science 課程包的資料。"""
    src = APP.read_text(encoding="utf-8")
    for word in ("Working", "Original", "雙表", "欄位與資料整合", "working.csv"):
        assert word not in src, word


def test_dual_start_page_has_no_missing_working_info_banner() -> None:
    src = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "目前還沒有 Working 工作資料" not in src
    assert "以下並排顯示雙表工作副本" not in src
    assert "答對後請 Agent 合併寫入 Original 與 Working" in src
    assert "請 Agent 對齊鍵名後合併" in src
