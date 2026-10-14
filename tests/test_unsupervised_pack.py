"""非監督式課程包（#13）：單獨安裝可用、課程包層 host 只有一層、套件依實際 import。"""

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

from add_dataset_streamlit_shell.course_packs import UNSUPERVISED, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES


def test_framework_plus_unsupervised_resolves_every_internal_import(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "unsupervised")

    files = installed_files(shell)
    assert "pages/11_Wards_Method.py" in files
    assert "pages/12_KMeans.py" in files
    assert "built-in-data/clustering/clustering_blobs_90.csv" in files
    assert unresolved_internal_imports(shell) == []
    assert sidebar(shell) == {
        "": ["總覽"],
        "機器學習 · 非監督式": ["Ward's Method（階層分群）", "K-Means 分群"],
    }


def _unsupervised_pack():
    from dataset_streamlit_shell.ui import unsupervised_pack

    return unsupervised_pack


def test_unsupervised_host_is_framework_plus_one_unsupervised_layer() -> None:
    from dataset_streamlit_shell.host_context import (
        assemble_host_context,
        framework_host_context,
    )
    from dataset_streamlit_shell.ui.data_science import data_science_pack_layer

    pack = _unsupervised_pack()
    layer = pack.unsupervised_pack_layer()
    host = pack.unsupervised_host_context()

    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1
    assert data_science_pack_layer() not in host
    assert "【教學頁】" not in host


def test_unsupervised_layer_uses_builtin_examples_and_writes_no_root_files() -> None:
    layer = _unsupervised_pack().unsupervised_pack_layer()

    assert "內建範例" in layer
    assert "built-in-data/clustering" in layer
    assert "可寫清單：無" in layer
    for root_file in ("original.csv", "working.csv", "ready.csv", "cleaning_log"):
        assert root_file not in layer  # 白名單：不點名其他課程包的檔
    # 不帶 Data Science 的整理指示
    assert "commit_dual_table_merge" not in layer
    assert "Working 工作資料路徑" not in layer


def test_unsupervised_chat_passes_unsupervised_host(monkeypatch) -> None:
    pack = _unsupervised_pack()
    captured: dict[str, object] = {}

    def fake_render_chat_panel(extra_context="", page_name="", **kwargs) -> None:
        captured.update(kwargs, extra_context=extra_context, page_name=page_name)

    monkeypatch.setattr(pack, "render_chat_panel", fake_render_chat_panel)

    pack.render_unsupervised_chat("目前切群數=3。", page_name="K-Means 分群")

    assert captured["host_context"] == pack.unsupervised_host_context()
    assert captured["extra_context"] == "目前切群數=3。"
    assert captured["page_name"] == "K-Means 分群"


def test_unsupervised_pages_open_chat_only_through_unsupervised_layer() -> None:
    for rel in pack_files(TEMPLATE_SHELL, UNSUPERVISED):
        if not rel.endswith(".py") or rel == "ui/unsupervised_pack.py":
            continue
        tree = ast.parse((TEMPLATE_SHELL / rel).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "render_chat_panel" not in names, rel
        assert "teaching_page_host_context" not in names, rel


def test_unsupervised_pack_owns_its_layer_module() -> None:
    assert "ui/unsupervised_pack.py" in pack_files(TEMPLATE_SHELL, UNSUPERVISED)


def test_unsupervised_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    runner = FakeDependencyRunner()
    shell = installed_shell(tmp_path, "unsupervised", runner=runner)
    module_to_dist = {
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "streamlit": "streamlit>=1.61",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, UNSUPERVISED))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    assert set(UNSUPERVISED.dependencies) == needed
    assert runner.calls[-1] == ["uv", "add", *UNSUPERVISED.dependencies]
