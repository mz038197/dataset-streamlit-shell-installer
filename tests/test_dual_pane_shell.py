"""Seams: dual_pane_shell clamp/constants (ADR-0006); dual-pane pages wire open_content_dual_pane."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "src" / "add_dataset_streamlit_shell" / "templates"
TEMPLATE = TEMPLATE_ROOT / "dataset_streamlit_shell"
UI = TEMPLATE / "ui"

if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui.dual_pane_shell import (  # noqa: E402
    AGENT_WIDTH_DEFAULT,
    AGENT_WIDTH_MAX,
    AGENT_WIDTH_MIN,
    AGENT_WIDTH_STORAGE_KEY,
    clamp_agent_width,
    pane_height_px,
)


def test_agent_width_constants_match_adr() -> None:
    assert AGENT_WIDTH_DEFAULT == 320
    assert AGENT_WIDTH_MIN == 260
    assert AGENT_WIDTH_MAX == 560
    assert AGENT_WIDTH_STORAGE_KEY == "dssAgentW"


def test_clamp_agent_width_bounds_and_rounding() -> None:
    assert clamp_agent_width(200) == 260
    assert clamp_agent_width(900) == 560
    assert clamp_agent_width(320) == 320
    assert clamp_agent_width(320.4) == 320
    assert clamp_agent_width(320.6) == 321


def test_pane_height_locks_to_viewport_minus_row_top() -> None:
    assert pane_height_px(viewport_height=1000, row_top=100, bottom_pad=8) == 892
    assert pane_height_px(viewport_height=300, row_top=200, bottom_pad=8) == 240


def test_dual_pane_pages_use_shared_shell_not_hardcoded_ratio() -> None:
    paths = [
        TEMPLATE / "app.py",
        TEMPLATE / "pages" / "2_Charts.py",
        UI / "workflow_ui.py",
        UI / "cv_layout.py",
        UI / "clustering_ui.py",
        UI / "cnn_ui.py",
        UI / "knn_ui.py",
        UI / "logistic_regression_ui.py",
        UI / "nn_ui.py",
        UI / "svm_ui.py",
        UI / "tree_ui.py",
    ]
    for path in paths:
        src = path.read_text(encoding="utf-8")
        assert "st.columns([5, 3]" not in src, path.name
        assert "open_content_dual_pane" in src, path.name


def test_chat_panel_marks_fill_height_host() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert 'data-dss-chat' in src
