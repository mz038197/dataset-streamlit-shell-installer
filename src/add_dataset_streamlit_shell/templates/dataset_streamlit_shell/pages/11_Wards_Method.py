from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import inject_style
from dataset_streamlit_shell.workflow_ui import pca_status, render_analysis_shell

st.set_page_config(page_title="Ward's Method", page_icon="WR", layout="wide")
inject_style()


def _render_wards(df: pd.DataFrame) -> None:
    status = pca_status(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("列數", f"{status['rows']:,}")
    c2.metric("數值欄位", f"{status['numeric_columns']:,}")
    c3.metric("數值缺失", f"{status['missing_cells']:,}")

    st.markdown("##### 適用情境")
    st.markdown(
        "- 沒有標籤欄位，想探索資料自然形成的群組。\n"
        "- 使用 Ward 最小變異法進行階層式分群，並以樹狀圖觀察合併過程。\n"
        "- 與 K-Means 不同：不必先指定群數 K，可先從樹狀圖決定切幾群。"
    )
    st.info("此頁面建置中。後續會加入欄位選擇、階層分群與樹狀圖。")

    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns]
    if len(numeric_columns) < 2:
        st.warning("階層分群至少需要 2 個數值欄位。請先完成欄位整理或編碼。")
        return

    st.markdown("##### 建議問 Agent")
    st.code(
        "請說明 Ward's Method 和 K-Means 分群的差異，以及什麼時候適合使用階層分群。",
        language="text",
    )


render_analysis_shell(
    "Ward's Method（階層分群）",
    "以 Ready 分析就緒資料進行 Ward 最小變異法的階層式分群。",
    _render_wards,
)
