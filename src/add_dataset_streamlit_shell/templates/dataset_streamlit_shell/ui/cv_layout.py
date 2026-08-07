from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from dataset_streamlit_shell.ui.data_ui import render_chat_panel
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane


def render_cv_tabbed_page(
    *,
    page_title: str,
    context_key: str,
    tab_labels: Sequence[str],
    tab_renderers: Sequence[Callable[[], None]],
) -> None:
    """Render CV teaching pages with one shared Agent panel beside tabbed main content."""
    teaching_col, agent_col = open_content_dual_pane()
    with agent_col:
        render_chat_panel(
            extra_context=str(
                st.session_state.get(context_key, f"目前頁面：{page_title}。")
            ),
            page_name=page_title,
        )
    with teaching_col:
        tabs = st.tabs(list(tab_labels))
        for tab, renderer in zip(tabs, tab_renderers, strict=True):
            with tab:
                renderer()
