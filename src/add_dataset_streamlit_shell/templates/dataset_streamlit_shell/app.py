from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.navigation import NavPage, installed_sections
from dataset_streamlit_shell.ui.data_ui import (
    ORIGINAL_DATASET_PATH,
    WORKING_DATASET_PATH,
    SHELL_ROOT,
    _display_path,
    brand_page_icon,
    inject_style,
    load_dataset,
    load_working_dataset,
    render_chat_panel,
    render_column_pills,
    render_dataset_metrics,
    working_dataset_file_exists,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane


st.set_page_config(page_title="資料學習實驗室", page_icon=brand_page_icon(), layout="wide")
inject_style()


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
            return

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
        render_chat_panel(page_name="總覽")


SECTIONS = {
    "": [
        NavPage(overview, "總覽", default=True),
    ],
    "AI 協作資料整理": [
        NavPage("pages/3_Field_Quality.py", "欄位與資料整合"),
        NavPage("pages/17_Data_Transform.py", "資料轉換"),
        NavPage("pages/4_Duplicates.py", "刪除重複資料列"),
        NavPage("pages/5_Numeric_Diagnostics.py", "缺失值處理"),
        NavPage("pages/6_Outliers.py", "離群值檢查"),
        NavPage("pages/7_Categorical.py", "類別欄位整理"),
        NavPage("pages/8_Encoding.py", "類別欄位編碼"),
        NavPage("pages/9_Correlation.py", "數值相關性"),
        NavPage("pages/13_Feature_Scaling.py", "特徵縮放"),
        NavPage("pages/2_Charts.py", "圖表探索"),
        NavPage("pages/20_Data_Split.py", "資料切分"),
    ],
    "機器學習 · 監督式": [
        NavPage("pages/14_Simple_Linear_Regression.py", "線性回歸"),
        NavPage("pages/16_Logistic_Regression.py", "邏輯迴歸"),
        NavPage("pages/18_Linear_SVM.py", "線性支持向量機"),
        NavPage("pages/19_KNN_Classification.py", "K-近鄰分類"),
        NavPage("pages/19_Decision_Tree_Concepts.py", "決策樹與集成"),
    ],
    "機器學習 · 非監督式": [
        NavPage("pages/11_Wards_Method.py", "Ward's Method（階層分群）"),
        NavPage("pages/12_KMeans.py", "K-Means 分群"),
    ],
    "深度學習": [
        NavPage("pages/21_Neural_Network.py", "類神經網路"),
    ],
    "電腦視覺": [
        NavPage("pages/27_CNN_Introduction.py", "卷積神經網路（CNN）"),
        NavPage("pages/22_Image_Classification.py", "影像分類（Image Classification）"),
        NavPage("pages/23_Object_Detection.py", "物件偵測（Object Detection）"),
        NavPage("pages/24_Semantic_Segmentation.py", "語意分割（Semantic Segmentation）"),
        NavPage("pages/25_Instance_Segmentation.py", "實例分割（Instance Segmentation）"),
        NavPage("pages/26_Promptable_Segmentation.py", "提示式分割（Promptable Segmentation / SAM）"),
    ],
    "降維分析": [
        NavPage("pages/10_PCA.py", "PCA 主成分分析"),
    ],
    "AI新創工作坊": [
        NavPage("pages/30_Startup_Challenge.py", "專案展示"),
    ],
}

pages = {
    section: [
        st.Page(
            page.source if callable(page.source) else str(SHELL_ROOT / page.source),
            title=page.title,
            default=page.default,
        )
        for page in section_pages
    ]
    for section, section_pages in installed_sections(SHELL_ROOT, SECTIONS).items()
}

st.navigation(pages).run()
