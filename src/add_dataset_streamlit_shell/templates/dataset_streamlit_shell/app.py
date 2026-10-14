from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.navigation import NavPage, installed_sections
from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    brand_page_icon,
    inject_style,
)
from dataset_streamlit_shell.ui.overview_ui import overview


st.set_page_config(page_title="資料學習實驗室", page_icon=brand_page_icon(), layout="wide")
inject_style()


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
