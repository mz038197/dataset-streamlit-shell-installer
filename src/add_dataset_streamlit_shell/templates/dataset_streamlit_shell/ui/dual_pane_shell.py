"""內容區雙欄殼：主教學欄｜資料 Agent 欄（ADR-0006）。"""

from __future__ import annotations

from typing import Any

import streamlit as st

AGENT_WIDTH_DEFAULT = 320
AGENT_WIDTH_MIN = 260
AGENT_WIDTH_MAX = 560
AGENT_WIDTH_STORAGE_KEY = "dssAgentW"


def clamp_agent_width(width: float | int) -> int:
    return max(AGENT_WIDTH_MIN, min(AGENT_WIDTH_MAX, int(round(float(width)))))


def pane_height_px(*, viewport_height: int, row_top: float, bottom_pad: int = 8) -> int:
    """Viewport-locked height for the dual-pane row (pure; used by chrome JS contract)."""
    return max(240, int(viewport_height - row_top - bottom_pad))


class _MarkedPane:
    def __init__(self, column: Any, pane: str) -> None:
        self._column = column
        self._pane = pane

    def __enter__(self) -> Any:
        result = self._column.__enter__()
        # st.html keeps data-* markers more reliably than markdown sanitization.
        st.html(f'<div data-dss-pane="{self._pane}" style="display:none"></div>')
        return result

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._column.__exit__(exc_type, exc, tb)


def inject_dual_pane_chrome() -> None:
    """Inject CSS/JS so teaching scrolls alone; Agent stays pinned; width drag-resizable."""
    css = f"""
<style>
  /* Do not scroll the app shell / sidebar when dual-pane markers exist. */
  html:has([data-dss-pane="main"]),
  body:has([data-dss-pane="main"]),
  .stApp:has([data-dss-pane="main"]),
  [data-testid="stAppViewContainer"]:has([data-dss-pane="main"]),
  [data-testid="stMain"]:has([data-dss-pane="main"]),
  section.main:has([data-dss-pane="main"]),
  .main:has([data-dss-pane="main"]),
  [data-testid="stAppScrollToBottomContainer"]:has([data-dss-pane="main"]) {{
    overflow: hidden !important;
    height: 100vh !important;
    max-height: 100vh !important;
  }}
  [data-testid="stMain"]:has([data-dss-pane="main"]) > div,
  section.main:has([data-dss-pane="main"]) > div.block-container {{
    height: 100% !important;
    max-height: 100% !important;
    overflow: hidden !important;
    padding-bottom: 0.5rem !important;
  }}
  /* First paint before JS: honour ADR default Agent width. */
  [data-testid="stColumn"]:has([data-dss-pane="agent"]) {{
    flex: 0 0 {AGENT_WIDTH_DEFAULT}px !important;
    width: {AGENT_WIDTH_DEFAULT}px !important;
    min-width: {AGENT_WIDTH_DEFAULT}px !important;
    max-width: {AGENT_WIDTH_DEFAULT}px !important;
  }}
  [data-testid="stColumn"]:has([data-dss-pane="main"]) {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
  }}
  .dss-dual-pane-row {{
    align-items: stretch !important;
    overflow: hidden !important;
    gap: 0.35rem !important;
  }}
  .dss-main-pane,
  .dss-main-pane > div,
  .dss-main-pane [data-testid="stVerticalBlock"] {{
    min-width: 0 !important;
  }}
  .dss-main-pane {{
    overflow-x: hidden !important;
    overflow-y: auto !important;
  }}
  .dss-agent-pane {{
    overflow: hidden !important;
    min-width: 0 !important;
  }}
  .dss-resizer {{
    flex: 0 0 5px !important;
    width: 5px !important;
    min-width: 5px !important;
    cursor: col-resize;
    align-self: stretch;
    background: transparent;
    border-radius: 999px;
  }}
  .dss-resizer:hover,
  body.dss-resizing .dss-resizer {{
    background: rgba(255, 75, 75, 0.35);
  }}
  body.dss-resizing {{
    cursor: col-resize !important;
    user-select: none !important;
  }}
</style>
"""
    js = f"""
<script>
(function () {{
  const KEY = {AGENT_WIDTH_STORAGE_KEY!r};
  const DEFAULT_W = {AGENT_WIDTH_DEFAULT};
  const MIN_W = {AGENT_WIDTH_MIN};
  const MAX_W = {AGENT_WIDTH_MAX};

  function clamp(w) {{
    w = Math.round(Number(w));
    if (!Number.isFinite(w)) return DEFAULT_W;
    return Math.max(MIN_W, Math.min(MAX_W, w));
  }}

  function readWidth() {{
    const raw = localStorage.getItem(KEY);
    if (raw == null || raw === "") return DEFAULT_W;
    return clamp(raw);
  }}

  function paneHeight(rowTop) {{
    return Math.max(240, Math.floor(window.innerHeight - rowTop - 8));
  }}

  function findPanes() {{
    const mainMark = document.querySelector('[data-dss-pane="main"]');
    const agentMark = document.querySelector('[data-dss-pane="agent"]');
    if (!mainMark || !agentMark) return null;
    const mainCol = mainMark.closest('[data-testid="stColumn"]');
    const agentCol = agentMark.closest('[data-testid="stColumn"]');
    if (!mainCol || !agentCol) return null;
    const row = mainCol.parentElement;
    if (!row || row !== agentCol.parentElement) return null;
    return {{ row, mainCol, agentCol }};
  }}

  function lockAppScroll() {{
    const nodes = document.querySelectorAll([
      "html",
      "body",
      ".stApp",
      '[data-testid="stAppViewContainer"]',
      '[data-testid="stMain"]',
      "section.main",
      ".main",
      '[data-testid="stAppScrollToBottomContainer"]',
    ].join(","));
    nodes.forEach(function (el) {{
      el.style.setProperty("overflow", "hidden", "important");
      el.style.setProperty("height", "100vh", "important");
      el.style.setProperty("max-height", "100vh", "important");
    }});
    document.querySelectorAll(
      '[data-testid="stMain"] .block-container, section.main > div.block-container'
    ).forEach(function (bc) {{
      bc.style.setProperty("overflow", "hidden", "important");
      bc.style.setProperty("height", "100%", "important");
      bc.style.setProperty("max-height", "100%", "important");
      bc.style.setProperty("padding-bottom", "0.5rem", "important");
    }});
  }}

  function rootVerticalBlock(col) {{
    return col.querySelector(':scope > div [data-testid="stVerticalBlock"], :scope [data-testid="stVerticalBlock"]');
  }}

  function pinColumn(col, h, scrollable) {{
    col.style.setProperty("height", h + "px", "important");
    col.style.setProperty("max-height", h + "px", "important");
    col.style.setProperty("overflow-x", "hidden", "important");
    col.style.setProperty("overflow-y", scrollable ? "auto" : "hidden", "important");
    const vb = rootVerticalBlock(col);
    if (vb) {{
      vb.style.setProperty("height", h + "px", "important");
      vb.style.setProperty("max-height", h + "px", "important");
      vb.style.setProperty("overflow-x", "hidden", "important");
      vb.style.setProperty("overflow-y", scrollable ? "auto" : "hidden", "important");
    }}
  }}

  function sizeChat(agentCol) {{
    const host = agentCol.querySelector("[data-dss-chat]");
    if (!host) return;
    let node = host.nextElementSibling;
    while (node && !node.matches('[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"]')) {{
      node = node.nextElementSibling;
    }}
    let target = null;
    if (node && node.matches('[data-testid="stVerticalBlockBorderWrapper"]')) {{
      target = node;
    }} else if (node) {{
      target = node.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
    }}
    if (!target) {{
      const wrappers = agentCol.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
      for (const el of wrappers) {{
        if (el.querySelector('[data-testid="stChatMessage"]') || (el.style && el.style.height)) {{
          target = el;
          break;
        }}
      }}
    }}
    if (!target) return;
    const top = target.getBoundingClientRect().top;
    const input = agentCol.querySelector('[data-testid="stChatInput"]');
    const bottomLimit = input
      ? input.getBoundingClientRect().top - 12
      : agentCol.getBoundingClientRect().bottom - 16;
    const avail = Math.floor(bottomLimit - top);
    if (avail >= 120) {{
      target.style.setProperty("height", avail + "px", "important");
      target.style.setProperty("max-height", avail + "px", "important");
      target.style.overflowY = "auto";
    }}
  }}

  function applyLayout(w) {{
    const panes = findPanes();
    if (!panes) return false;
    lockAppScroll();
    const {{ row, mainCol, agentCol }} = panes;
    row.classList.add("dss-dual-pane-row");
    mainCol.classList.add("dss-main-pane");
    agentCol.classList.add("dss-agent-pane");

    const h = paneHeight(row.getBoundingClientRect().top);
    row.style.setProperty("height", h + "px", "important");
    row.style.setProperty("max-height", h + "px", "important");
    row.style.setProperty("overflow", "hidden", "important");
    row.style.setProperty("align-items", "stretch", "important");

    agentCol.style.setProperty("flex", "0 0 " + w + "px", "important");
    agentCol.style.setProperty("width", w + "px", "important");
    agentCol.style.setProperty("min-width", w + "px", "important");
    agentCol.style.setProperty("max-width", w + "px", "important");
    mainCol.style.setProperty("flex", "1 1 0", "important");
    mainCol.style.setProperty("min-width", "0", "important");
    mainCol.style.removeProperty("width");
    mainCol.style.removeProperty("max-width");

    pinColumn(mainCol, h, true);
    pinColumn(agentCol, h, false);

    let handle = row.querySelector(":scope > .dss-resizer");
    if (!handle) {{
      handle = document.createElement("div");
      handle.className = "dss-resizer";
      handle.title = "拖曳調整資料 Agent 欄寬度";
      row.insertBefore(handle, agentCol);
      handle.addEventListener("mousedown", function (e) {{
        e.preventDefault();
        document.body.classList.add("dss-resizing");
        const onMove = function (ev) {{
          const rect = row.getBoundingClientRect();
          const next = clamp(rect.right - ev.clientX);
          localStorage.setItem(KEY, String(next));
          applyLayout(next);
        }};
        const onUp = function () {{
          document.body.classList.remove("dss-resizing");
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        }};
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      }});
    }} else if (handle.nextElementSibling !== agentCol) {{
      row.insertBefore(handle, agentCol);
    }}

    sizeChat(agentCol);
    return true;
  }}

  function boot() {{
    if (applyLayout(readWidth())) return;
    let tries = 0;
    const timer = setInterval(function () {{
      tries += 1;
      if (applyLayout(readWidth()) || tries > 60) clearInterval(timer);
    }}, 50);
  }}

  boot();
  window.addEventListener("resize", function () {{
    applyLayout(readWidth());
  }});
  if (!window.__dssDualPaneObs) {{
    let debounce = null;
    window.__dssDualPaneObs = new MutationObserver(function () {{
      clearTimeout(debounce);
      debounce = setTimeout(function () {{ applyLayout(readWidth()); }}, 40);
    }});
    window.__dssDualPaneObs.observe(document.body, {{ childList: true, subtree: true }});
  }}
}})();
</script>
"""
    st.html(css + js, unsafe_allow_javascript=True)


def open_content_dual_pane() -> tuple[_MarkedPane, _MarkedPane]:
    """Open 主教學欄 / 資料 Agent 欄 with shared dual-pane chrome."""
    inject_dual_pane_chrome()
    teaching, agent = st.columns([1, 1], gap="small")
    return _MarkedPane(teaching, "main"), _MarkedPane(agent, "agent")
