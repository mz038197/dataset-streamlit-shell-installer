from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import (
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
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane


st.set_page_config(page_title="資料學習實驗室", page_icon="CSV", layout="wide")
inject_style()


def overview() -> None:
    teaching, agent = open_content_dual_pane()

    with teaching:
        st.title("資料學習實驗室")
        st.caption(
            "從雙表起點經資料整合建立 Working，再透過 Agent 協作整理，建立 Ready 分析就緒資料。"
        )

        source_df = load_dataset()
        working_df = load_working_dataset()
        ready_df = load_ready_dataset()
        df = working_df if working_df is not None else source_df
        if df is None:
            st.info(
                "尚未建立工作資料。請到「欄位與資料概覽」查看乘客表／航程表，"
                "再到「資料整合」合併後會建立 Original 原始資料與 Working 工作資料。"
            )
            return

        st.markdown('<div class="data-card">', unsafe_allow_html=True)
        render_dataset_metrics(df)
        st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("##### 資料生命週期")
        st.write(
            "Original 原始資料：資料整合套用合併時寫入，只作為重置來源，不直接修改。"
        )
        st.write("Working 工作資料：Agent 協作整理與診斷的主要工作區。")
        st.write(
            "Ready 分析就緒資料：整理完成後凍結，供圖表探索、降維等分析頁使用；"
            "監督式與非監督式教學頁（回歸／分類／K-Means／Ward's Method 等）使用各頁內建範例資料。"
        )
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
        st.dataframe(df.head(12), width="stretch", hide_index=True)

        st.markdown("##### 課程流程")
        st.markdown(
            """
1. 先開「欄位與資料概覽」看乘客表／航程表，再到「資料整合」合併，寫入 Original／Working。
2. 經「資料轉換」與後續清理診斷 `working.csv`，請右側 Agent 一步一步整理。
3. 在「建立 Ready 分析就緒資料」產生 `ready.csv`，再到圖表探索；需要時用「資料切分」寫出 train／val／test。
4. 監督式與非監督式教學頁仍使用各頁內建範例資料。
"""
        )

    with agent:
        render_chat_panel(page_name="總覽")


pages = {
    "": [
        st.Page(overview, title="總覽", default=True),
    ],
    "AI 協作資料整理": [
        st.Page(str(SHELL_ROOT / "pages" / "3_Field_Quality.py"), title="欄位與資料概覽"),
        st.Page(str(SHELL_ROOT / "pages" / "15_Data_Integration.py"), title="資料整合"),
        st.Page(str(SHELL_ROOT / "pages" / "17_Data_Transform.py"), title="資料轉換"),
        st.Page(str(SHELL_ROOT / "pages" / "4_Duplicates.py"), title="刪除重複資料列"),
        st.Page(str(SHELL_ROOT / "pages" / "5_Numeric_Diagnostics.py"), title="缺失值處理"),
        st.Page(str(SHELL_ROOT / "pages" / "6_Outliers.py"), title="離群值檢查"),
        st.Page(str(SHELL_ROOT / "pages" / "7_Categorical.py"), title="類別欄位整理"),
        st.Page(str(SHELL_ROOT / "pages" / "8_Encoding.py"), title="類別欄位編碼"),
        st.Page(str(SHELL_ROOT / "pages" / "9_Correlation.py"), title="數值相關性"),
        st.Page(str(SHELL_ROOT / "pages" / "13_Feature_Scaling.py"), title="特徵縮放"),
        st.Page(str(SHELL_ROOT / "pages" / "8_Ready.py"), title="建立 Ready 分析就緒資料"),
        st.Page(str(SHELL_ROOT / "pages" / "2_Charts.py"), title="圖表探索"),
        st.Page(str(SHELL_ROOT / "pages" / "20_Data_Split.py"), title="資料切分"),
    ],
    "機器學習 · 監督式": [
        st.Page(str(SHELL_ROOT / "pages" / "14_Simple_Linear_Regression.py"), title="線性回歸"),
        st.Page(str(SHELL_ROOT / "pages" / "16_Logistic_Regression.py"), title="邏輯迴歸"),
        st.Page(str(SHELL_ROOT / "pages" / "18_Linear_SVM.py"), title="線性支持向量機"),
        st.Page(str(SHELL_ROOT / "pages" / "19_KNN_Classification.py"), title="K-近鄰分類"),
        st.Page(str(SHELL_ROOT / "pages" / "19_Decision_Tree_Concepts.py"), title="決策樹與集成"),
    ],
    "機器學習 · 非監督式": [
        st.Page(str(SHELL_ROOT / "pages" / "11_Wards_Method.py"), title="Ward's Method（階層分群）"),
        st.Page(str(SHELL_ROOT / "pages" / "12_KMeans.py"), title="K-Means 分群"),
    ],
    "深度學習": [
        st.Page(str(SHELL_ROOT / "pages" / "21_Neural_Network.py"), title="類神經網路"),
    ],
    "電腦視覺": [
        st.Page(
            str(SHELL_ROOT / "pages" / "27_CNN_Introduction.py"),
            title="卷積神經網路（CNN）",
        ),
        st.Page(
            str(SHELL_ROOT / "pages" / "22_Image_Classification.py"),
            title="影像分類（Image Classification）",
        ),
        st.Page(
            str(SHELL_ROOT / "pages" / "23_Object_Detection.py"),
            title="物件偵測（Object Detection）",
        ),
        st.Page(
            str(SHELL_ROOT / "pages" / "24_Semantic_Segmentation.py"),
            title="語意分割（Semantic Segmentation）",
        ),
        st.Page(
            str(SHELL_ROOT / "pages" / "25_Instance_Segmentation.py"),
            title="實例分割（Instance Segmentation）",
        ),
        st.Page(
            str(SHELL_ROOT / "pages" / "26_Promptable_Segmentation.py"),
            title="提示式分割（Promptable Segmentation / SAM）",
        ),
    ],
    "降維分析": [
        st.Page(str(SHELL_ROOT / "pages" / "10_PCA.py"), title="PCA 主成分分析"),
    ],
}

st.navigation(pages).run()
