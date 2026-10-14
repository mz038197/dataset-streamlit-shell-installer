from __future__ import annotations

from collections.abc import Callable, Sequence

import streamlit as st

from dataset_streamlit_shell.ui.computer_vision_pack import render_computer_vision_chat
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
        render_computer_vision_chat(page_title=page_title, context_key=context_key)
    with teaching_col:
        tabs = st.tabs(list(tab_labels))
        for tab, renderer in zip(tabs, tab_renderers, strict=True):
            with tab:
                renderer()
