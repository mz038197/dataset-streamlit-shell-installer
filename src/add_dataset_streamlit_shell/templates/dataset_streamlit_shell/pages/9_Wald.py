from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import inject_style
from dataset_streamlit_shell.workflow_ui import render_analysis_shell, wald_status

st.set_page_config(page_title="Wald 法", page_icon="WA", layout="wide")
inject_style()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _render_wald(df: pd.DataFrame) -> None:
    status = wald_status(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("列數", f"{status['rows']:,}")
    c2.metric("數值欄位", f"{status['numeric_columns']:,}")
    c3.metric("缺失儲存格", f"{status['missing_cells']:,}")

    st.markdown("##### 適用情境")
    st.markdown(
        "- 二元比例：例如存活率、通過率、是否購買。\n"
        "- 平均數估計：例如平均年齡、平均票價。\n"
        "- 迴歸係數：本階段先作為概念說明，完整模型留到後續課程。"
    )

    mode = st.radio("Wald 檢查類型", ["二元比例", "平均數"], horizontal=True)
    if mode == "二元比例":
        binary_columns = status["binary_columns"]
        if not binary_columns:
            st.warning("目前沒有剛好兩個值的欄位。請先完成資料整理或選擇其他分析方式。")
            return
        column = st.selectbox("二元欄位", binary_columns)
        values = sorted(df[column].dropna().unique().tolist())
        success = st.selectbox("成功/目標類別", values)
        sample = df[column].dropna()
        n = len(sample)
        p_hat = float((sample == success).mean()) if n else 0.0
        se = math.sqrt(p_hat * (1 - p_hat) / n) if n else 0.0
        lower = p_hat - 1.96 * se
        upper = p_hat + 1.96 * se
        st.metric("估計比例", f"{p_hat:.3f}")
        st.write(f"95% Wald 信賴區間：約 [{lower:.3f}, {upper:.3f}]")
    else:
        numeric_columns = [str(c) for c in df.select_dtypes(include="number").columns]
        if not numeric_columns:
            st.warning("平均數 Wald 檢查需要至少一個數值欄位。")
            return
        column = st.selectbox("數值欄位", numeric_columns)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        n = len(values)
        mean = float(values.mean()) if n else 0.0
        se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        lower = mean - 1.96 * se
        upper = mean + 1.96 * se
        st.metric("平均數估計", f"{mean:.3f}")
        st.write(f"95% Wald 信賴區間：約 [{lower:.3f}, {upper:.3f}]")

    st.markdown("##### 建議問 Agent")
    st.code("請根據目前 Ready 分析就緒資料，說明 Wald 法在這個欄位上的適用性與限制。", language="text")
    st.code("請解釋剛剛的 Wald 信賴區間代表什麼，並指出樣本數是否足夠。", language="text")
    st.caption(f"標準常態 1.96 對應累積機率約 {_normal_cdf(1.96):.3f}。")


render_analysis_shell("Wald 法", "以 Ready 分析就緒資料進行比例或平均數的 Wald 概念檢查。", _render_wald)
