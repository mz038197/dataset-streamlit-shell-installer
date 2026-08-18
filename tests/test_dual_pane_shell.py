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


def test_dual_pane_chrome_clears_overlay_header() -> None:
    chrome = (UI / "dual_pane_shell.py").read_text(encoding="utf-8")
    styles = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "padding-top: 4rem !important" in chrome
    assert 'setProperty("padding-top", "4rem"' in chrome
    assert "padding-top: 4rem !important" in styles
    assert "padding-left: 1rem !important" in styles
    assert "padding-right: 1rem !important" in styles
    assert "padding-left: 1rem !important" in chrome
    assert "padding-right: 1rem !important" in chrome
    assert 'setProperty("padding-left", "1rem"' in chrome
    assert 'setProperty("padding-right", "1rem"' in chrome
    assert "0.5rem" not in styles.split("padding-top", 1)[1][:40]
    assert '[data-testid="stHtml"] style' in chrome
    assert '[data-testid="stElementContainer"]:has([data-dss-pane])' in chrome
    assert ".dss-main-pane [data-testid=\"stHeading\"]" in chrome
    assert "data-agent-title-spacer" not in styles
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    shell = context.split("**內容區雙欄殼**:", 1)[1].split("**", 1)[0]
    assert "1rem" in shell
    assert "只縮一邊內邊距" in shell


def test_sidebar_brand_uses_classroom_logo_and_wordmark() -> None:
    styles = (UI / "data_ui.py").read_text(encoding="utf-8")
    app = (TEMPLATE / "app.py").read_text(encoding="utf-8")
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    logo = TEMPLATE / "assets" / "brand-logo.png"
    sidebar_block = styles.split('[data-testid="stSidebarHeader"]', 1)[1]
    wordmark = sidebar_block.split("::after", 1)[1].split("}", 1)[0]
    assert logo.is_file()
    assert "st.logo" in styles
    assert 'size="small"' in styles
    assert '[data-testid="stSidebarHeader"]::after' in styles
    assert '[data-testid="stLogo"]::after' not in styles
    assert 'content: "VansCoding.AI"' in wordmark
    assert '[data-testid="stHeader"]' not in styles
    assert "40px" in sidebar_block
    assert "#eef5f5" in sidebar_block
    assert "#007070" in wordmark
    assert "#009999" in wordmark
    assert "#f8c000" in wordmark
    assert "background-clip: text" in wordmark
    assert "font-weight: 700" in wordmark
    assert "font-size: 14px" in wordmark
    assert "letter-spacing: -0.02em" in wordmark
    assert "overflow: visible" in sidebar_block
    collapse = sidebar_block.split("stSidebarCollapseButton", 1)[1].split("}", 1)[0]
    assert "margin-left: auto" in collapse
    assert "order: 2" in collapse
    assert '[data-testid="stSidebarHeader"] button' not in styles
    assert "brand_page_icon" in styles
    assert "page_icon=brand_page_icon()" in app
    assert "**側欄品牌列**:" in context


def test_agent_chat_pins_input_and_scrolls_messages() -> None:
    """資料 Agent 欄：訊息自捲、chat_input 釘欄底（對齊 waku dock）。"""
    chrome = (UI / "dual_pane_shell.py").read_text(encoding="utf-8")
    panel = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "layoutAgentChat" in chrome
    assert "findChatInput" in chrome
    assert "pinBottomChatToAgent" in chrome
    assert "margin-top" in chrome and "auto" in chrome
    assert "Math.max(0, avail)" in chrome
    assert "scrollTop" in chrome
    # Must not use absolute pin on stChatInput (nested containing blocks clip it).
    assert ".dss-agent-pane [data-testid=\"stChatInput\"]" not in chrome
    assert "st.container(height=240" in panel
    # border=False container has no BorderWrapper; grow the height=240 scrollport itself.
    assert "data-test-scroll-behavior" in chrome
    assert "DOCUMENT_POSITION_FOLLOWING" in chrome
    assert "dss-chat-scroll" in chrome
    # st.container(height=N) also locks parent stLayoutWrapper to flex: 0 0 Npx.
    assert "chatScrollHost" in chrome
    assert "dss-chat-scroll-host" in chrome
    assert "flex-basis" in chrome
    # Host is LayoutWrapper only; top/height must be measured from the scroll target.
    host_fn = chrome.split("function chatScrollHost", 1)[1].split(
        "function clearInnerHeightLocks", 1
    )[0]
    assert "|| el.parentElement" not in host_fn
    assert "target.getBoundingClientRect().top" in chrome
    assert "topEl" not in chrome.split("function layoutAgentChat", 1)[1].split(
        "function applyLayout", 1
    )[0]


def test_chat_panel_has_no_image_attachment_path() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "附加圖片" not in src
    assert "_save_uploaded_chat_image" not in src
    assert "CHAT_IMAGE_DIR" not in src
    assert "file_uploader" not in src
    assert "image_path=" not in src or "image_path=None" in src


def test_chat_panel_tts_stays_collapsed_expander() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert 'st.expander("語音播放", expanded=False)' in src


def test_content_dual_pane_glossary_pins_agent_input() -> None:
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    block = context.split("**內容區雙欄殼**:", 1)[1].split("**", 1)[0]
    assert "釘" in block
    assert "訊息" in block or "聊天" in block
    assert "附圖" in block or "附加圖片" in block
