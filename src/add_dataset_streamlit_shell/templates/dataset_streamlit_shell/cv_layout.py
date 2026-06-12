from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from dataset_streamlit_shell.data_ui import render_chat_panel


def render_cv_tabbed_page(
    *,
    page_title: str,
    context_key: str,
    tab_labels: Sequence[str],
    tab_renderers: Sequence[Callable[[], None]],
) -> None:
    """Render CV teaching pages with one shared Agent panel beside tabbed main content."""
    main_col, side_col = st.columns([5, 3], gap="large")
    with side_col:
        render_chat_panel(
            extra_context=str(
                st.session_state.get(context_key, f"目前頁面：{page_title}。")
            ),
            page_name=page_title,
        )
    with main_col:
        tabs = st.tabs(list(tab_labels))
        for tab, renderer in zip(tabs, tab_renderers, strict=True):
            with tab:
                renderer()
