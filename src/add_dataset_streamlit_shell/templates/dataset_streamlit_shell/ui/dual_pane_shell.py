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
  /* Lock page scroll; teaching column becomes the scrollport. */
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
  [data-testid="stMain"] .block-container:has([data-dss-pane="main"]),
  section.main > div.block-container:has([data-dss-pane="main"]) {{
    max-height: 100% !important;
    overflow: hidden !important;
    padding-top: 4rem !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }}
  /* Chrome 的 st.html 列不要把雙欄往下推。 */
  [data-testid="stElementContainer"]:has([data-testid="stHtml"] style) {{
    display: none !important;
  }}
  /* Pane marker 內層 display:none 仍佔一個 element slot + 欄內 gap。 */
  [data-testid="stElementContainer"]:has([data-dss-pane]) {{
    display: none !important;
  }}
  /* 欄頂貼齊第一個標題，不要再疊 st.title 預設 padding-top。 */
  .dss-main-pane [data-testid="stHeading"],
  .dss-main-pane [data-testid="stHeading"] h1,
  .dss-agent-pane [data-testid="stHeading"],
  .dss-agent-pane [data-testid="stHeading"] h1 {{
    margin-top: 0 !important;
    padding-top: 0 !important;
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
    min-height: 0 !important;
  }}
  /*
   * Flex items default min-height:auto (= content size), which prevents
   * max-height from creating an internal scrollbar — content gets clipped
   * by an overflow:hidden ancestor instead. Force min-height:0.
   */
  .dss-main-pane,
  .dss-main-pane > div,
  .dss-agent-pane,
  .dss-agent-pane > div {{
    min-width: 0 !important;
    min-height: 0 !important;
  }}
  .dss-main-pane {{
    overflow-x: hidden !important;
    overflow-y: auto !important;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
  }}
  .dss-agent-pane {{
    overflow: hidden !important;
    position: relative !important;
  }}
  /* Keep Streamlit page-bottom chat host visible if input lands there. */
  [data-testid="stBottom"] {{
    z-index: 40 !important;
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
      bc.style.setProperty("max-height", "100%", "important");
      bc.style.setProperty("padding-top", "4rem", "important");
      bc.style.setProperty("padding-bottom", "0.5rem", "important");
      bc.style.setProperty("padding-left", "1rem", "important");
      bc.style.setProperty("padding-right", "1rem", "important");
    }});
  }}

  function isAgentChatScrollport(vb) {{
    /* st.container(height=...) scrollport; layoutAgentChat owns its height. */
    return (
      vb
      && vb.getAttribute("data-testid") === "stVerticalBlock"
      && (vb.hasAttribute("data-test-scroll-behavior") || vb.classList.contains("dss-chat-scroll"))
    );
  }}

  function chatScrollHost(el) {{
    /* st.container(height=N) wraps the scrollport in stLayoutWrapper with
       emotion flex: 0 0 Npx — that host must grow too or height sticks at N.
       Return only that wrapper (or null); never a random parentElement. */
    if (!el) return null;
    return el.closest('[data-testid="stLayoutWrapper"]');
  }}

  function clearInnerHeightLocks(col) {{
    /* Column is the scrollport; do not lock nested vertical blocks to a fixed
       height (that clipped siblings / quiz blocks below the fold).
       Keep the Agent chat list scrollport — clearing it left height stuck at
       Streamlit's placeholder 240px with no working grow target. */
    col.querySelectorAll('[data-testid="stVerticalBlock"]').forEach(function (vb) {{
      if (col.classList.contains("dss-agent-pane") && isAgentChatScrollport(vb)) return;
      vb.style.removeProperty("height");
      vb.style.removeProperty("max-height");
      vb.style.setProperty("min-height", "0", "important");
      vb.style.removeProperty("overflow-y");
      vb.style.removeProperty("overflow");
    }});
    Array.prototype.forEach.call(col.children, function (child) {{
      if (child.classList && child.classList.contains("dss-chat-scroll-host")) return;
      child.style.setProperty("min-height", "0", "important");
      child.style.removeProperty("height");
      child.style.removeProperty("max-height");
    }});
  }}

  function pinColumn(col, h, scrollable) {{
    col.style.setProperty("height", h + "px", "important");
    col.style.setProperty("max-height", h + "px", "important");
    col.style.setProperty("min-height", "0", "important");
    col.style.setProperty("overflow-x", "hidden", "important");
    col.style.setProperty("overflow-y", scrollable ? "auto" : "hidden", "important");
    clearInnerHeightLocks(col);
  }}

  function findChatScrollTarget(agentCol) {{
    const host = agentCol.querySelector("[data-dss-chat]");
    if (!host) return null;

    /* border=False: height lives on stVerticalBlock, not BorderWrapper.
       data-dss-chat is nested, so host.nextElementSibling is usually null. */
    const scrollBlocks = agentCol.querySelectorAll(
      '[data-testid="stVerticalBlock"][data-test-scroll-behavior]'
    );
    for (const el of scrollBlocks) {{
      if (host.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) {{
        return el;
      }}
    }}

    let probe = host.closest('[data-testid="stElementContainer"]') || host;
    let node = probe.nextElementSibling;
    while (node) {{
      if (node.matches('[data-testid="stVerticalBlockBorderWrapper"]')) return node;
      const wrap = node.querySelector('[data-testid="stVerticalBlockBorderWrapper"]');
      if (wrap) return wrap;
      if (isAgentChatScrollport(node)) return node;
      const nested = node.querySelector('[data-testid="stVerticalBlock"][data-test-scroll-behavior]');
      if (nested) return nested;
      node = node.nextElementSibling;
    }}

    const wrappers = agentCol.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
    for (const el of wrappers) {{
      if (el.querySelector('[data-testid="stChatMessage"]') || (el.style && el.style.height)) {{
        return el;
      }}
    }}
    return null;
  }}

  function findChatInput(agentCol) {{
    return (
      agentCol.querySelector('[data-testid="stChatInput"]')
      || document.querySelector('[data-testid="stBottom"] [data-testid="stChatInput"]')
    );
  }}

  function pinBottomChatToAgent(agentCol, input) {{
    /* st.chat_input sometimes lands in page-bottom host; dock it to Agent column. */
    const bottom = input.closest('[data-testid="stBottom"]');
    if (!bottom) return false;
    const colRect = agentCol.getBoundingClientRect();
    const bottomGap = Math.max(0, window.innerHeight - colRect.bottom);
    bottom.style.setProperty("position", "fixed", "important");
    bottom.style.setProperty("left", colRect.left + "px", "important");
    bottom.style.setProperty("width", colRect.width + "px", "important");
    bottom.style.setProperty("right", "auto", "important");
    bottom.style.setProperty("bottom", bottomGap + "px", "important");
    bottom.style.setProperty("z-index", "40", "important");
    bottom.style.setProperty("display", "block", "important");
    bottom.style.setProperty("visibility", "visible", "important");
    bottom.style.setProperty("opacity", "1", "important");
    return true;
  }}

  function layoutAgentChat(agentCol) {{
    /*
     * Keep chat_input in normal flow (absolute + nested position:relative parents
     * was clipping it under overflow:hidden). Size the message list to the space
     * above the input; push the input to the column bottom with margin-top:auto.
     */
    const colRect = agentCol.getBoundingClientRect();
    const input = findChatInput(agentCol);
    const pinnedBottom = input ? pinBottomChatToAgent(agentCol, input) : false;
    const inputH = input
      ? Math.max(56, Math.ceil(input.getBoundingClientRect().height))
      : 72;
    const pad = 8;
    const target = findChatScrollTarget(agentCol);

    if (input && !pinnedBottom) {{
      input.style.removeProperty("position");
      input.style.removeProperty("left");
      input.style.removeProperty("right");
      input.style.removeProperty("bottom");
      const wrap =
        input.closest('[data-testid="stElementContainer"]') || input.parentElement;
      if (wrap) {{
        wrap.style.setProperty("margin-top", "auto", "important");
        wrap.style.setProperty("flex-shrink", "0", "important");
        wrap.style.setProperty("position", "relative", "important");
        wrap.style.setProperty("z-index", "6", "important");
      }}
      const rootVb = agentCol.querySelector('[data-testid="stVerticalBlock"]');
      if (rootVb) {{
        rootVb.style.setProperty("display", "flex", "important");
        rootVb.style.setProperty("flex-direction", "column", "important");
        rootVb.style.setProperty("height", "100%", "important");
        rootVb.style.setProperty("min-height", "0", "important");
      }}
    }}

    if (!target) return;
    target.classList.add("dss-chat-scroll");
    const host = chatScrollHost(target);
    /* Always measure from the scrollport itself — host may be absent. */
    const top = target.getBoundingClientRect().top;
    const avail = Math.floor(colRect.bottom - top - inputH - pad);
    const height = Math.max(0, avail);
    if (host && host !== target) {{
      host.classList.add("dss-chat-scroll-host");
      host.style.setProperty("height", height + "px", "important");
      host.style.setProperty("max-height", height + "px", "important");
      host.style.setProperty("flex", "1 1 auto", "important");
      host.style.setProperty("flex-basis", "auto", "important");
      host.style.setProperty("min-height", "0", "important");
      host.style.setProperty("min-width", "0", "important");
      host.style.setProperty("overflow", "hidden", "important");
    }}
    target.style.setProperty("height", height + "px", "important");
    target.style.setProperty("max-height", height + "px", "important");
    target.style.setProperty("flex", "1 1 auto", "important");
    target.style.setProperty("min-height", "0", "important");
    target.style.setProperty("overflow-y", "auto", "important");
    const gap = target.scrollHeight - target.scrollTop - target.clientHeight;
    if (!target.dataset.dssStick || gap < 80) {{
      target.scrollTop = target.scrollHeight;
      target.dataset.dssStick = "1";
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
    row.style.setProperty("min-height", "0", "important");
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

    layoutAgentChat(agentCol);
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
