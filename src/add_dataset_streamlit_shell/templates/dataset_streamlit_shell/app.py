from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import (
    DATASET_PATH,
    FILTERED_DATASET_PATH,
    SHELL_ROOT,
    _display_path,
    inject_style,
    load_dataset,
    render_chat_panel,
    render_column_pills,
    render_dataset_metrics,
)


st.set_page_config(page_title="CSV Data Agent Shell", page_icon="CSV", layout="wide")
inject_style()


def overview() -> None:
    main, side = st.columns([5, 3], gap="large")

    with main:
        st.title("CSV Data Agent Shell")
        st.caption(
            "上傳任意 CSV、篩選目前工作資料，將完成的 `agent_core.Agent` 套到右側聊天框。"
        )

        df = load_dataset()
        if df is None:
            st.info("請到 Database 頁上傳 CSV。上傳後會寫入目前 shell 的 `data/current.csv`。")
            return

        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        render_dataset_metrics(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### DATASET")
        st.write("完整資料：目前資料集")
        st.write("工作資料：Agent 工作資料")
        with st.expander("技術資訊", expanded=False):
            st.caption(f"完整資料檔：`{_display_path(DATASET_PATH)}`")
            st.caption(f"Agent 工作資料檔：`{_display_path(FILTERED_DATASET_PATH)}`")
        render_column_pills(df.columns)

        st.markdown("##### QUICK PREVIEW")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)

        st.markdown("##### LESSON FLOW")
        st.markdown(
            """
1. 在 Database 頁上傳或更換 CSV。
2. 用欄位選擇與欄位值篩選調整畫面檢視；這些篩選不會覆蓋 CSV。
3. 完成 WG-22 後，右側 Agent 會透過 `Agent.chat(..., on_token=...)` 分析同一份 CSV。
4. 若請 Agent 補值、計算或新增欄位，目標工作檔是 `current_filtered.csv`。
"""
        )

    with side:
        render_chat_panel()


st.navigation(
    [
        st.Page(overview, title="Overview", default=True),
        st.Page(str(SHELL_ROOT / "pages" / "1_Database.py"), title="Database"),
        st.Page(str(SHELL_ROOT / "pages" / "2_Charts.py"), title="Charts"),
    ]
).run()
