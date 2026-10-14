from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import (
    SHELL_ROOT,
    brand_page_icon,
    inject_style,
)
from dataset_streamlit_shell.ui.overview_ui import overview


st.set_page_config(page_title="資料學習實驗室", page_icon=brand_page_icon(), layout="wide")
inject_style()


pages = {
    "": [
        st.Page(overview, title="總覽", default=True),
    ],
    "AI 協作資料整理": [
        st.Page(str(SHELL_ROOT / "pages" / "3_Field_Quality.py"), title="欄位與資料整合"),
        st.Page(str(SHELL_ROOT / "pages" / "17_Data_Transform.py"), title="資料轉換"),
        st.Page(str(SHELL_ROOT / "pages" / "4_Duplicates.py"), title="刪除重複資料列"),
        st.Page(str(SHELL_ROOT / "pages" / "5_Numeric_Diagnostics.py"), title="缺失值處理"),
        st.Page(str(SHELL_ROOT / "pages" / "6_Outliers.py"), title="離群值檢查"),
        st.Page(str(SHELL_ROOT / "pages" / "7_Categorical.py"), title="類別欄位整理"),
        st.Page(str(SHELL_ROOT / "pages" / "8_Encoding.py"), title="類別欄位編碼"),
        st.Page(str(SHELL_ROOT / "pages" / "9_Correlation.py"), title="數值相關性"),
        st.Page(str(SHELL_ROOT / "pages" / "13_Feature_Scaling.py"), title="特徵縮放"),
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
    "AI新創工作坊": [
        st.Page(
            str(SHELL_ROOT / "pages" / "30_Startup_Challenge.py"),
            title="專案展示",
        ),
    ],
}

st.navigation(pages).run()
