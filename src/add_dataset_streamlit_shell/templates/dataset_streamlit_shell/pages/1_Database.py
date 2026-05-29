from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.data_ui import (
    FILTERED_DATASET_PATH,
    _display_path,
    initialize_working_dataset,
    inject_style,
    load_dataset,
    load_working_dataset,
    read_uploaded_csv,
    render_chat_panel,
    render_column_pills,
    render_dataset_metrics,
    reset_working_dataset,
    save_dataset,
)


st.set_page_config(page_title="CSV Database", page_icon="DB", layout="wide")
inject_style()


def _date_range_filter(series: pd.Series) -> tuple[date, date] | None:
    parsed = pd.to_datetime(series, errors="coerce")
    valid = parsed.dropna()
    if valid.empty:
        return None
    if len(valid) / max(len(series.dropna()), 1) < 0.8:
        return None
    return valid.min().date(), valid.max().date()


def _apply_value_filter(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, Any]:
    series = df[column]
    numeric = pd.to_numeric(series, errors="coerce")
    non_null_count = int(series.notna().sum())
    numeric_count = int(numeric.notna().sum())

    if non_null_count and numeric_count / non_null_count >= 0.8:
        values = numeric.dropna()
        if values.empty:
            return df, ("number", None)
        min_value = float(values.min())
        max_value = float(values.max())
        if min_value == max_value:
            st.caption(f"`{column}` 只有單一數值：{min_value:g}")
            return df, ("number", min_value, max_value, True)
        selected = st.slider(
            f"{column} range",
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
            key=f"filter_number_{column}",
        )
        keep_blank = st.checkbox(
            f"保留 `{column}` 空值",
            value=True,
            key=f"filter_number_null_{column}",
        )
        mask = (numeric >= selected[0]) & (numeric <= selected[1])
        if keep_blank:
            mask = mask | numeric.isna()
        return df[mask], ("number", selected[0], selected[1], keep_blank)

    date_range = _date_range_filter(series)
    if date_range is not None:
        selected_dates = st.date_input(
            f"{column} date range",
            value=date_range,
            key=f"filter_date_{column}",
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start, end = selected_dates
            parsed = pd.to_datetime(series, errors="coerce")
            mask = parsed.isna() | (
                (parsed.dt.date >= start) & (parsed.dt.date <= end)
            )
            return df[mask], ("date", start.isoformat(), end.isoformat())
        return df, ("date", None)

    text_values = sorted(series.dropna().astype(str).unique().tolist())
    if len(text_values) <= 200:
        selected = st.multiselect(
            f"{column} values",
            text_values,
            default=text_values,
            key=f"filter_text_{column}",
        )
        if selected:
            return df[series.astype(str).isin(selected)], ("text", tuple(selected))
        return df.iloc[0:0], ("text", ())

    search = st.text_input(
        f"{column} contains",
        placeholder="輸入關鍵字篩選文字欄位",
        key=f"filter_search_{column}",
    )
    if search:
        return df[series.astype(str).str.contains(search, case=False, na=False)], (
            "search",
            search,
        )
    return df, ("search", "")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("篩選條件", expanded=True):
        all_columns = [str(column) for column in df.columns]
        selected_columns = st.multiselect(
            "顯示欄位",
            all_columns,
            default=all_columns,
            key="selected_columns",
        )

        filter_columns = st.multiselect(
            "依欄位值篩選",
            selected_columns,
            default=[],
            key="filter_columns",
        )

        filtered = df.copy()
        for column in filter_columns:
            st.markdown(f"###### `{column}`")
            filtered, filter_state = _apply_value_filter(filtered, column)

    if not selected_columns:
        st.warning("至少選擇一個顯示欄位。")
        return filtered.iloc[:, 0:0]
    return filtered[selected_columns]


main, side = st.columns([5, 3], gap="large")

with main:
    st.title("Database")
    st.caption(
        "上傳 CSV 後，中間表格會呈現目前工作資料；篩選條件只影響畫面，不會覆蓋 Agent 工作資料。"
    )

    uploaded = st.file_uploader("上傳 CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = read_uploaded_csv(uploaded)
        except Exception as exc:
            st.error(f"CSV 讀取失敗：{exc}")
            st.stop()
        save_dataset(df)
        reset_working_dataset()
        initialize_working_dataset(df)
        st.success(
            "已上傳並建立完整資料 `current.csv` 與工作資料 `current_filtered.csv`。"
        )

    df = load_working_dataset()
    if df is None:
        st.info("尚未載入資料。請上傳 CSV。")
    else:
        render_dataset_metrics(df)
        st.markdown("##### COLUMNS")
        render_column_pills(df.columns)

        filtered = apply_filters(df)

        st.markdown("##### TABLE")
        st.caption(
            f"目前顯示 {len(filtered):,} / {len(df):,} 筆；這是畫面篩選結果，不會寫回 CSV。"
        )
        with st.expander("技術資訊", expanded=False):
            if FILTERED_DATASET_PATH.exists():
                st.caption(f"目前基底資料：Agent 工作資料 `{_display_path(FILTERED_DATASET_PATH)}`")
            else:
                st.caption("目前基底資料：完整資料 `dataset_streamlit_shell/data/current.csv`")
            st.caption("畫面篩選不會落檔；只有右側 Agent 會寫入 Agent 工作資料。")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        with st.expander("欄位統計", expanded=False):
            st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

with side:
    render_chat_panel()
