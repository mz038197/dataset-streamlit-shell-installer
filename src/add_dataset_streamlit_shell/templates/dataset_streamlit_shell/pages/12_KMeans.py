from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import inject_style
from dataset_streamlit_shell.workflow_ui import pca_status, render_analysis_shell

st.set_page_config(page_title="K-Means 分群", page_icon="KM", layout="wide")
inject_style()


def _render_kmeans(df) -> None:
    status = pca_status(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("列數", f"{status['rows']:,}")
    c2.metric("數值欄位", f"{status['numeric_columns']:,}")
    c3.metric("數值缺失", f"{status['missing_cells']:,}")

    st.markdown("##### 適用情境")
    st.markdown(
        "- 沒有標籤欄位，想先把資料分成 K 個群。\n"
        "- 需要指定群數 K，觀察各群中心與樣本分布。\n"
        "- 常與 PCA 搭配：先降維再分群，方便視覺化。"
    )
    st.info("此頁面建置中。後續會加入欄位選擇、K 值設定與分群結果視覺化。")

    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]
    if len(numeric_columns) < 2:
        st.warning("K-Means 分群至少需要 2 個數值欄位。請先完成欄位整理或編碼。")
        return

    st.markdown("##### 建議問 Agent")
    st.code(
        "請說明 K-Means 分群前為什麼常需要標準化，以及 K 值可以怎麼初步選擇。",
        language="text",
    )


render_analysis_shell(
    "K-Means 分群",
    "以 Ready 分析就緒資料進行 K-Means 分群。",
    _render_kmeans,
)
