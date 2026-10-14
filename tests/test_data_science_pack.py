"""Data Science 課程包層 host 與每輪 Working 摘要（ADR 0030）。"""

from __future__ import annotations

import ast
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

from add_dataset_streamlit_shell.course_packs import DATA_SCIENCE, pack_files  # noqa: E402
from dataset_streamlit_shell.host_context import (  # noqa: E402
    assemble_host_context,
    framework_host_context,
)
from dataset_streamlit_shell.ui import data_science  # noqa: E402

SHELL_ROOT = TEMPLATE_ROOT / "dataset_streamlit_shell"


def test_data_science_host_is_framework_plus_data_science_layer() -> None:
    layer = data_science.data_science_pack_layer()
    host = data_science.data_science_host_context()
    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1


def test_data_science_layer_lists_writable_files() -> None:
    layer = data_science.data_science_pack_layer()
    assert "可寫清單" in layer
    for path in (
        "dataset_streamlit_shell/workspace/working.csv",
        "dataset_streamlit_shell/workspace/original.csv",
        "dataset_streamlit_shell/workspace/integration/passengers.csv",
        "dataset_streamlit_shell/workspace/integration/voyage.csv",
        "dataset_streamlit_shell/workspace/cleaning_log.jsonl",
    ):
        assert path in layer, path


def test_data_science_layer_keeps_prep_rules_without_framework_sentences() -> None:
    layer = data_science.data_science_pack_layer()
    assert "commit_dual_table_merge" in layer
    assert "passenger_id" in layer and "PassengerId" in layer
    assert "ready.csv" in layer
    assert "created_at、actor、action、columns、rows、note" in layer
    assert "教學頁" not in layer
    # 讀檔習慣與腳本位置已在框架層
    assert "read_file" not in layer
    assert "scripts" not in layer


class _Session(dict):
    def pop(self, key, default=None):
        return dict.pop(self, key, default)


def _capture_chat(monkeypatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_render_chat_panel(extra_context="", page_name="", **kwargs) -> None:
        captured.update(kwargs, extra_context=extra_context, page_name=page_name)

    monkeypatch.setattr(data_science, "render_chat_panel", fake_render_chat_panel)
    monkeypatch.setattr(data_science.st, "session_state", _Session())
    return captured


def test_data_science_chat_passes_ds_host_and_working_summary(tmp_path, monkeypatch) -> None:
    captured = _capture_chat(monkeypatch)
    working = tmp_path / "working.csv"
    working.write_text("Age,Fare\n1,2\n3,4\n5,6\n", encoding="utf-8")
    monkeypatch.setattr(data_science, "WORKING_DATASET_PATH", working)
    monkeypatch.setattr(data_science, "SPLIT_DATASET_PATHS", ())
    monkeypatch.setattr(data_science, "READY_DATASET_PATH", tmp_path / "ready.csv")

    data_science.render_data_science_chat("資料切分頁。", page_name="資料切分")

    assert captured["host_context"] == data_science.data_science_host_context()
    assert captured["page_name"] == "資料切分"
    snapshot = captured["extra_context"]
    text = snapshot() if callable(snapshot) else snapshot
    assert "工作資料目前有 3 筆、2 欄" in text
    assert "資料切分頁。" in text


def test_data_science_chat_without_working_hints_dual_tables(tmp_path, monkeypatch) -> None:
    captured = _capture_chat(monkeypatch)
    monkeypatch.setattr(data_science, "WORKING_DATASET_PATH", tmp_path / "working.csv")
    monkeypatch.setattr(data_science, "ORIGINAL_DATASET_PATH", tmp_path / "original.csv")
    monkeypatch.setattr(data_science, "SPLIT_DATASET_PATHS", ())
    monkeypatch.setattr(data_science, "READY_DATASET_PATH", tmp_path / "ready.csv")

    data_science.render_data_science_chat(page_name="欄位與資料整合")

    snapshot = captured["extra_context"]
    text = snapshot() if callable(snapshot) else snapshot
    assert "尚未建立工作資料" in text


def test_data_science_pages_open_chat_only_through_data_science_layer() -> None:
    """DS 課程包內的檔案不直接呼叫框架聊天欄，一律經 render_data_science_chat。"""
    for rel in pack_files(SHELL_ROOT, DATA_SCIENCE):
        if not rel.endswith(".py") or rel == "ui/data_science.py":
            continue
        tree = ast.parse((SHELL_ROOT / rel).read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "render_chat_panel" not in called, rel
