"""降維分析課程包（#13）：單獨安裝可用、自帶整理線副本、沒有 Working 時提示。"""

from __future__ import annotations

import ast
from pathlib import Path

from pack_install_helpers import (
    TEMPLATE_SHELL,
    FakeDependencyRunner,
    installed_files,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import DATA_SCIENCE, PCA, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES


def test_framework_plus_pca_resolves_every_internal_import(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "pca")

    files = installed_files(shell)
    assert "pages/10_PCA.py" in files
    assert unresolved_internal_imports(shell) == []
    assert sidebar(shell) == {"": ["總覽"], "降維分析": ["PCA 主成分分析"]}


def test_pca_installs_no_data_science_files(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "pca")

    files = installed_files(shell)
    for owned in DATA_SCIENCE.paths:
        assert not any(rel == owned or rel.startswith(owned + "/") for rel in files), owned


def test_pca_pack_shares_no_file_with_data_science() -> None:
    assert not set(pack_files(TEMPLATE_SHELL, PCA)) & set(
        pack_files(TEMPLATE_SHELL, DATA_SCIENCE)
    )


def test_pca_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()
    shell = installed_shell(tmp_path, "pca", runner=runner)
    module_to_dist = {
        "numpy": "numpy",
        "pandas": "pandas",
        "streamlit": "streamlit>=1.61",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, PCA))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    assert set(PCA.dependencies) == needed
    assert runner.calls[-1] == ["uv", "add", *PCA.dependencies]


def _pca_pack():
    from dataset_streamlit_shell.ui import pca_pack

    return pca_pack


def test_pca_hint_asks_to_add_data_science_when_working_is_missing(tmp_path: Path) -> None:
    hint = _pca_pack().pca_missing_working_hint(tmp_path / "working.csv")

    assert hint is not None
    assert "請先加裝 Data Science 課程包並建立 Working 工作資料" in hint


def test_pca_has_no_hint_when_working_exists(tmp_path: Path) -> None:
    working = tmp_path / "working.csv"
    working.write_text("a,b\n1,2\n", encoding="utf-8")

    assert _pca_pack().pca_missing_working_hint(working) is None


def test_pca_host_is_framework_plus_one_pca_layer() -> None:
    from dataset_streamlit_shell.host_context import (
        assemble_host_context,
        framework_host_context,
    )

    pack = _pca_pack()
    layer = pack.pca_pack_layer()
    host = pack.pca_host_context()

    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1
    assert host.count("課程包】") == 1
    assert "【教學頁】" not in host


def test_pca_layer_is_a_data_science_layer_copy_with_working_whitelist() -> None:
    layer = _pca_pack().pca_pack_layer()

    assert "可寫清單" in layer
    assert "dataset_streamlit_shell/workspace/working.csv" in layer
    assert "dataset_streamlit_shell/workspace/cleaning_log.jsonl" in layer
    assert "created_at、actor、action、columns、rows、note" in layer
    assert "不要建立 ready.csv" in layer
    assert "PCA" in layer


class _Session(dict):
    def pop(self, key, default=None):
        return dict.pop(self, key, default)


def _capture_chat(monkeypatch, tmp_path: Path) -> dict[str, object]:
    pack = _pca_pack()
    captured: dict[str, object] = {}

    def fake_render_chat_panel(extra_context="", page_name="", **kwargs) -> None:
        captured.update(kwargs, extra_context=extra_context, page_name=page_name)

    monkeypatch.setattr(pack, "render_chat_panel", fake_render_chat_panel)
    monkeypatch.setattr(pack.st, "session_state", _Session())
    monkeypatch.setattr(pack, "WORKING_DATASET_PATH", tmp_path / "working.csv")
    monkeypatch.setattr(pack, "READY_DATASET_PATH", tmp_path / "ready.csv")
    monkeypatch.setattr(pack, "SPLIT_DATASET_PATHS", ())
    return captured


def _turn_text(captured: dict[str, object]) -> str:
    snapshot = captured["extra_context"]
    return snapshot() if callable(snapshot) else snapshot


def test_pca_chat_passes_pca_host_and_working_summary(tmp_path, monkeypatch) -> None:
    captured = _capture_chat(monkeypatch, tmp_path)
    (tmp_path / "working.csv").write_text("Age,Fare\n1,2\n3,4\n5,6\n", encoding="utf-8")

    _pca_pack().render_pca_chat(page_name="PCA 主成分分析")

    assert captured["host_context"] == _pca_pack().pca_host_context()
    assert captured["page_name"] == "PCA 主成分分析"
    assert "工作資料目前有 3 筆、2 欄" in _turn_text(captured)


def test_pca_chat_without_working_carries_the_hint(tmp_path, monkeypatch) -> None:
    captured = _capture_chat(monkeypatch, tmp_path)

    _pca_pack().render_pca_chat(page_name="PCA 主成分分析")

    assert "請先加裝 Data Science 課程包" in _turn_text(captured)


def test_pca_status_counts_numeric_columns() -> None:
    import pandas as pd

    status = _pca_pack().pca_status(
        pd.DataFrame({"a": [1.0, None, 3.0], "b": [4, 5, 6], "c": ["x", "y", "z"]})
    )

    assert status["rows"] == 3
    assert status["numeric_columns"] == 2
    assert status["missing_cells"] == 1
    assert status["enough_columns"] is True


def test_pca_pages_open_chat_only_through_pca_layer() -> None:
    for rel in pack_files(TEMPLATE_SHELL, PCA):
        if not rel.endswith(".py") or rel == "ui/pca_pack.py":
            continue
        tree = ast.parse((TEMPLATE_SHELL / rel).read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "render_chat_panel" not in imported, rel
        assert "teaching_page_host_context" not in imported, rel
