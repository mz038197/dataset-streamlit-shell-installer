from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import inject_style
from dataset_streamlit_shell.ui.workflow_ui import pca_status, render_analysis_shell

st.set_page_config(page_title="PCA 主成分分析", page_icon="PC", layout="wide")
inject_style()


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)


def _render_pca(df: pd.DataFrame) -> None:
    status = pca_status(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("列數", f"{status['rows']:,}")
    c2.metric("數值欄位", f"{status['numeric_columns']:,}")
    c3.metric("數值缺失", f"{status['missing_cells']:,}")

    numeric_columns = [str(c) for c in df.select_dtypes(include="number").columns]
    if len(numeric_columns) < 2:
        st.warning("PCA 至少需要 2 個數值欄位。請先完成欄位整理或編碼。")
        return

    selected = st.multiselect(
        "選擇 PCA 欄位",
        numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
    )
    if len(selected) < 2:
        st.warning("請至少選擇 2 個數值欄位。")
        return

    working = df[selected].apply(pd.to_numeric, errors="coerce").dropna()
    if len(working) < 3:
        st.warning("可用樣本少於 3 筆，暫不適合 PCA。")
        return

    standardize = st.checkbox("標準化欄位", value=True)
    matrix = _standardize(working) if standardize else working
    matrix = matrix.dropna(axis=1, how="any")
    if matrix.shape[1] < 2:
        st.warning("標準化後可用欄位少於 2 個。")
        return

    centered = matrix - matrix.mean()
    _, singular_values, vt = np.linalg.svd(centered.to_numpy(), full_matrices=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    scores = centered.to_numpy() @ vt.T

    st.markdown("##### 解釋變異量")
    variance_frame = pd.DataFrame(
        {
            "主成分": [f"PC{i + 1}" for i in range(len(explained))],
            "解釋變異比例": explained,
        }
    )
    st.dataframe(variance_frame.head(10), width="stretch", hide_index=True)
    st.bar_chart(variance_frame.set_index("主成分")["解釋變異比例"])

    if scores.shape[1] >= 2:
        st.markdown("##### PC1 / PC2 散點圖")
        plot_frame = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1]})
        st.scatter_chart(plot_frame, x="PC1", y="PC2")

    st.markdown("##### 欄位權重")
    loading = pd.DataFrame(vt[:2].T, index=matrix.columns, columns=["PC1", "PC2"])
    st.dataframe(loading.style.format("{:.3f}"), width="stretch")

    st.markdown("##### 建議問 Agent")
    st.code("請解讀解釋變異量，說明前兩個主成分保留了多少資訊。", language="text")
    st.code("請根據欄位權重表格，說明哪些欄位最影響 PC1 與 PC2。", language="text")
    st.code("請檢查目前 Ready 分析就緒資料是否還需要標準化或處理缺失值後再做 PCA。", language="text")


render_analysis_shell("PCA 主成分分析", "以 Ready 分析就緒資料進行主成分分析與條件檢查。", _render_pca)
