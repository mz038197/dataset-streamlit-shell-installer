from __future__ import annotations

import streamlit as st

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ui.data_ui import (
    ORIGINAL_DATASET_PATH,
    WORKING_DATASET_PATH,
    _display_path,
    load_dataset,
    load_working_dataset,
    render_chat_panel,
    render_column_pills,
    render_dataset_metrics,
    working_dataset_file_exists,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane


def overview_host_context() -> str:
    """總覽只用框架層，不疊任何課程包層。"""
    return assemble_host_context()


def overview() -> None:
    teaching, agent = open_content_dual_pane()

    with teaching:
        st.title("資料學習實驗室")
        st.caption(
            "從雙表起點經資料整合建立 Working，再透過 Agent 協作整理。"
            "圖表探索、資料切分與 PCA 都讀 Working 工作資料。"
        )

        source_df = load_dataset()
        working_df = load_working_dataset()
        has_working = working_dataset_file_exists()
        df = working_df if working_df is not None else source_df
        if df is None:
            st.info(
                "尚未建立工作資料。請到「欄位與資料整合」查看乘客表／航程表，"
                "請右側 Agent 對齊鍵名並合併後會建立 Original 原始資料與 Working 工作資料。"
            )
        else:
            with st.container(key="overview_data_card"):
                render_dataset_metrics(df)

            st.divider()
            st.markdown("##### 資料生命週期")
            st.write(
                "Original 原始資料：資料整合寫入時與 Working 同批建立，只作為重置來源，不直接修改。"
            )
            st.write("Working 工作資料：Agent 協作整理與診斷的主要工作區；圖表探索、PCA、資料切分都讀這份。")
            st.write(
                "監督式與非監督式教學頁（回歸／分類／K-Means／Ward's Method 等）使用各頁內建範例資料。"
            )
            with st.expander("技術資訊", expanded=False):
                st.caption(f"Original 原始資料檔：`{_display_path(ORIGINAL_DATASET_PATH)}`")
                st.caption(f"Working 工作資料檔：`{_display_path(WORKING_DATASET_PATH)}`")
            render_column_pills(df.columns)

            if has_working and working_df is not None:
                st.download_button(
                    "下載 Working 工作資料",
                    data=working_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="working.csv",
                    mime="text/csv",
                    width="stretch",
                )

            st.markdown("##### 快速預覽")
            st.dataframe(df.head(12), width="stretch", hide_index=True)

            st.markdown("##### 課程流程")
            st.markdown(
                """
1. 先開「欄位與資料整合」看乘客表／航程表，請 Agent 對齊鍵名後合併，寫入 Original／Working。
2. 經「資料轉換」與後續清理診斷 `working.csv`，請右側 Agent 一步一步整理。
3. 圖表探索與 PCA 直接讀 Working；需要時用「資料切分」寫出 train／val／test。
4. 監督式與非監督式教學頁仍使用各頁內建範例資料。
"""
            )

    with agent:
        render_chat_panel(
            page_name="總覽",
            host_context=overview_host_context(),
            skip_working_snapshot=True,
        )
