from __future__ import annotations

import streamlit as st

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ui.data_ui import render_chat_panel
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane


def overview_host_context() -> str:
    """總覽只用框架層，不疊任何課程包層。"""
    return assemble_host_context()


def overview() -> None:
    """總覽只介紹 AI Lab 框架，不讀任何課程包的資料檔。"""
    teaching, agent = open_content_dual_pane()

    with teaching:
        st.title("資料學習實驗室")
        st.caption("AI Lab：左邊是教學頁，右邊是陪你一起做的資料 Agent。")

        st.markdown("##### 怎麼使用")
        st.markdown(
            """
1. 在右側按「啟用資料 Agent」，就可以用中文向 Agent 提問。
2. 對話紀錄會自動保存，可用右上角的選單切換、新增或刪除對話。
3. Agent 只會在各頁允許的範圍內讀檔、寫檔；需要腳本時會放在 `scripts` 資料夾。
"""
        )

        st.markdown("##### 課程包")
        st.write(
            "每一段課程是一個課程包，上到哪一段就依老師說明加裝哪一包；"
            "加裝後重新整理頁面，左側導覽就會出現該段的教學頁。"
        )

    with agent:
        render_chat_panel(
            page_name="總覽",
            host_context=overview_host_context(),
        )
