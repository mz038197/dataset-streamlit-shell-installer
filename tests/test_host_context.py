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

from dataset_streamlit_shell.host_context import (  # noqa: E402
    assemble_host_context,
    framework_host_context,
)

PACK_LAYER = "【課程包層】只寫 workspace/example.csv。"
PAGE_LAYER = "【頁面層】寫回協定。"


def test_assemble_orders_framework_then_pack_then_page() -> None:
    host = assemble_host_context(PACK_LAYER, PAGE_LAYER)
    framework = framework_host_context()
    assert host.startswith(framework)
    assert host.index(framework) < host.index(PACK_LAYER) < host.index(PAGE_LAYER)
    assert host.endswith(PAGE_LAYER)


def test_assemble_without_page_layer_ends_with_pack_layer() -> None:
    host = assemble_host_context(PACK_LAYER)
    assert host.startswith(framework_host_context())
    assert host.endswith(PACK_LAYER)


def test_assemble_without_pack_layer_is_framework_only() -> None:
    assert assemble_host_context() == framework_host_context()
    assert assemble_host_context(None) == framework_host_context()


PACK_VOCABULARY = (
    "Working",
    "working.csv",
    "Original",
    "original.csv",
    "雙表",
    "cleaning_log",
    "挑戰",
    "Challenge",
    "challenge",
    "ready.csv",
    "內建範例",
    "Titanic",
    "鐵達尼",
)


def test_framework_layer_has_no_course_pack_vocabulary() -> None:
    text = framework_host_context()
    for word in PACK_VOCABULARY:
        assert word not in text, word


def test_framework_layer_states_role_read_first_scripts_and_whitelist() -> None:
    text = framework_host_context()
    assert "資料 Agent" in text
    assert "read_file" in text
    assert "dataset_streamlit_shell/scripts" in text
    assert "可寫清單" in text


def test_overview_host_is_framework_layer_only() -> None:
    from dataset_streamlit_shell.ui.overview_ui import overview_host_context

    assert overview_host_context() == framework_host_context()
