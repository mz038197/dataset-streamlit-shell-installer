"""監督式課程包（#12）：只裝「框架 + supervised」的安裝結果與 host 組裝。"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pack_install_helpers import (
    TEMPLATE_SHELL,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import SUPERVISED, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES

from dataset_streamlit_shell.host_context import (
    assemble_host_context,
    framework_host_context,
)
from dataset_streamlit_shell.ui import supervised_pack

SUPERVISED_TITLES = ["線性回歸", "邏輯迴歸", "線性支持向量機", "K-近鄰分類", "決策樹與集成"]


def test_framework_plus_supervised_resolves_every_internal_import(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "supervised")

    assert unresolved_internal_imports(shell) == []


def test_framework_plus_supervised_sidebar_shows_overview_and_supervised(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "supervised")

    assert sidebar(shell) == {"": ["總覽"], "機器學習 · 監督式": SUPERVISED_TITLES}


def test_supervised_pack_owns_its_pack_layer_module() -> None:
    assert "ui/supervised_pack.py" in pack_files(TEMPLATE_SHELL, SUPERVISED)


def test_supervised_dependencies_match_its_third_party_imports(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "supervised")
    module_to_dist = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "streamlit": "streamlit>=1.61",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, SUPERVISED))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    assert set(SUPERVISED.dependencies) == needed


# --- host context：框架層 + 監督式課程包層（+ 頁面層） ---


def test_supervised_layer_uses_builtin_examples_and_has_no_writable_root_files() -> None:
    layer = supervised_pack.supervised_pack_layer()

    assert "內建範例資料" in layer
    assert "dataset_streamlit_shell/built-in-data/regression" in layer
    assert "dataset_streamlit_shell/built-in-data/classification" in layer
    assert "可寫清單" in layer
    # 白名單寫法：不再點名 Working／Original 等其他課程包的檔案
    for word in ("Working", "Original", "working.csv", "original.csv", "雙表", "cleaning_log"):
        assert word not in layer, word
    # 讀檔習慣、腳本位置屬框架層
    assert "read_file" not in layer
    assert "scripts" not in layer


def test_supervised_host_is_framework_plus_supervised_layer() -> None:
    layer = supervised_pack.supervised_pack_layer()

    host = supervised_pack.supervised_host_context()

    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1


def test_supervised_host_appends_page_layer_after_pack_layer() -> None:
    layer = supervised_pack.supervised_pack_layer()

    host = supervised_pack.supervised_host_context("【某頁】寫回協定")

    assert host == assemble_host_context(layer, "【某頁】寫回協定")


SUPERVISED_UI_MODULES = ("lr_ui", "logistic_regression_ui", "svm_ui", "knn_ui", "tree_ui")


@pytest.mark.parametrize("module_name", SUPERVISED_UI_MODULES)
def test_page_host_has_framework_and_exactly_one_pack_layer(module_name: str) -> None:
    module = importlib.import_module(f"dataset_streamlit_shell.ui.{module_name}")
    layer = supervised_pack.supervised_pack_layer()

    host = module.page_host_context()

    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1
    assert host.count("課程包】") == 1
    assert "【教學頁】" not in host
    assert "working.csv" not in host and "original.csv" not in host


@pytest.mark.parametrize("module_name", ("svm_ui", "knn_ui", "tree_ui"))
def test_pages_without_write_back_get_no_page_layer(module_name: str) -> None:
    module = importlib.import_module(f"dataset_streamlit_shell.ui.{module_name}")

    assert module.page_host_context() == supervised_pack.supervised_host_context()


@pytest.mark.parametrize(
    ("module_name", "page_marker", "slots_file", "request_file"),
    [
        ("lr_ui", "【線性回歸頁】", "lr_slots.json", "lr_train_request.json"),
        (
            "logistic_regression_ui",
            "【邏輯迴歸頁】",
            "logistic_slots.json",
            "logistic_train_request.json",
        ),
    ],
)
def test_write_back_pages_pass_their_protocol_as_page_layer(
    module_name: str, page_marker: str, slots_file: str, request_file: str
) -> None:
    module = importlib.import_module(f"dataset_streamlit_shell.ui.{module_name}")
    layer = supervised_pack.supervised_pack_layer()

    host = module.page_host_context()

    page_layer = host.split(layer, 1)[1]
    assert page_layer.lstrip().startswith(page_marker)
    assert f"dataset_streamlit_shell/workspace/{slots_file}" in page_layer
    assert f"dataset_streamlit_shell/workspace/{request_file}" in page_layer
