from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import (
    ORIGINAL_DATASET_PATH,
    READY_DATASET_PATH,
    WORKING_DATASET_PATH,
    SHELL_ROOT,
    _display_path,
    inject_style,
    load_dataset,
    load_ready_dataset,
    load_working_dataset,
    render_chat_panel,
    render_column_pills,
    render_dataset_metrics,
)


st.set_page_config(page_title="資料學習實驗室", page_icon="CSV", layout="wide")
inject_style()


def overview() -> None:
    main, side = st.columns([5, 3], gap="large")

    with main:
        st.title("資料學習實驗室")
        st.caption(
            "上傳 CSV，透過 Agent 協作整理 Working 工作資料，建立 Ready 分析就緒資料。"
        )

        source_df = load_dataset()
        working_df = load_working_dataset()
        ready_df = load_ready_dataset()
        df = working_df if working_df is not None else source_df
        if df is None:
            st.info("請到「資料上傳與預覽」頁上傳 CSV。上傳後會建立 Original 原始資料與 Working 工作資料。")
            return

        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        render_dataset_metrics(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### 資料生命週期")
        st.write("Original 原始資料：上傳後保留，不直接修改。")
        st.write("Working 工作資料：Agent 協作整理與診斷的主要工作區。")
        st.write("Ready 分析就緒資料：整理完成後凍結，供後續學習、分析與訓練使用。")
        with st.expander("技術資訊", expanded=False):
            st.caption(f"Original 原始資料檔：`{_display_path(ORIGINAL_DATASET_PATH)}`")
            st.caption(f"Working 工作資料檔：`{_display_path(WORKING_DATASET_PATH)}`")
            st.caption(f"Ready 分析就緒資料檔：`{_display_path(READY_DATASET_PATH)}`")
        render_column_pills(df.columns)

        if ready_df is None:
            st.warning("尚未建立 Ready 分析就緒資料。完成資料整理後，請到「建立 Ready 分析就緒資料」頁產生 `ready.csv`。")
        else:
            st.success(f"Ready 分析就緒資料已建立：{len(ready_df):,} 筆、{len(ready_df.columns):,} 欄。")

        st.markdown("##### 快速預覽")
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)

        st.markdown("##### 課程流程")
        st.markdown(
            """
1. 在「資料上傳與預覽」上傳或更換 CSV。
2. 到「AI 協作資料整理」診斷 `working.csv`，請右側 Agent 一步一步整理資料。
3. 在「建立 Ready 分析就緒資料」產生 `ready.csv`。
4. 後續學習頁面預設讀取 `ready.csv`。
"""
        )

    with side:
        render_chat_panel(page_name="總覽")


pages = {
    "資料工作區": [
        st.Page(overview, title="總覽", default=True),
        st.Page(str(SHELL_ROOT / "pages" / "1_Database.py"), title="資料上傳與預覽"),
        st.Page(str(SHELL_ROOT / "pages" / "2_Charts.py"), title="通用圖表"),
    ],
    "AI 協作資料整理": [
        st.Page(str(SHELL_ROOT / "pages" / "3_Field_Quality.py"), title="欄位與資料概覽"),
        st.Page(str(SHELL_ROOT / "pages" / "4_Duplicates.py"), title="刪除重複資料列"),
        st.Page(str(SHELL_ROOT / "pages" / "5_Numeric_Diagnostics.py"), title="缺失值處理"),
        st.Page(str(SHELL_ROOT / "pages" / "6_Outliers.py"), title="離群值檢查"),
        st.Page(str(SHELL_ROOT / "pages" / "7_Categorical.py"), title="類別欄位整理"),
        st.Page(str(SHELL_ROOT / "pages" / "8_Encoding.py"), title="類別欄位編碼"),
        st.Page(str(SHELL_ROOT / "pages" / "9_Correlation.py"), title="數值相關性"),
        st.Page(str(SHELL_ROOT / "pages" / "13_Feature_Scaling.py"), title="特徵縮放"),
        st.Page(str(SHELL_ROOT / "pages" / "8_Ready.py"), title="建立 Ready 分析就緒資料"),
    ],
    "降維分析": [
        st.Page(str(SHELL_ROOT / "pages" / "10_PCA.py"), title="PCA 主成分分析"),
    ],
    "機器學習 · 監督式": [
        st.Page(str(SHELL_ROOT / "pages" / "14_Simple_Linear_Regression.py"), title="單變量線性回歸"),
        st.Page(str(SHELL_ROOT / "pages" / "15_Multiple_Linear_Regression.py"), title="多變量線性回歸"),
        st.Page(str(SHELL_ROOT / "pages" / "16_Logistic_Regression.py"), title="邏輯迴歸"),
        st.Page(str(SHELL_ROOT / "pages" / "17_Regularized_Logistic_Regression.py"), title="正則化邏輯迴歸"),
        st.Page(str(SHELL_ROOT / "pages" / "18_Linear_SVM.py"), title="線性 SVM"),
        st.Page(str(SHELL_ROOT / "pages" / "19_Decision_Tree_Concepts.py"), title="決策樹概念"),
        st.Page(str(SHELL_ROOT / "pages" / "20_XGBoost.py"), title="XGBoost"),
    ],
    "機器學習 · 非監督式": [
        st.Page(str(SHELL_ROOT / "pages" / "11_Wards_Method.py"), title="Ward's Method（階層分群）"),
        st.Page(str(SHELL_ROOT / "pages" / "12_KMeans.py"), title="K-Means 分群"),
    ],
    "深度學習": [
        st.Page(str(SHELL_ROOT / "pages" / "21_Neural_Network.py"), title="類神經網路"),
    ],
}

st.navigation(pages).run()
