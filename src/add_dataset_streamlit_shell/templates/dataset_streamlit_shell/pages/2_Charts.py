from __future__ import annotations

import io
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_streamlit_shell.ui.data_ui import (
    ORIGINAL_DATASET_PATH,
    READY_DATASET_PATH,
    WORKING_DATASET_PATH,
    _display_path,
    inject_style,
    load_dataset,
    load_ready_dataset,
    render_chat_panel,
    render_dataset_metrics,
)
from dataset_streamlit_shell.plotting import configure_matplotlib_for_traditional_chinese


st.set_page_config(page_title="通用圖表", page_icon="CH", layout="wide")
inject_style()

COUNT_ROWS = "資料筆數"
ChartType = Literal["bar", "pie", "stacked_bar", "line", "radar", "histogram"]
Aggregation = Literal["count", "sum", "mean", "median"]
configure_matplotlib_for_traditional_chinese()


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _all_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns]


def _load_chart_dataset(source: str) -> tuple[pd.DataFrame | None, str, Path | None, str | None]:
    if source == "Ready 分析就緒資料":
        dataset = load_ready_dataset()
        return (
            dataset,
            "Ready 分析就緒資料",
            READY_DATASET_PATH if dataset is not None else None,
            None if dataset is not None else "尚未建立 Ready 分析就緒資料，請先到「建立 Ready 分析就緒資料」頁產生 ready.csv。",
        )

    if source == "Working 工作資料":
        if WORKING_DATASET_PATH.exists():
            return (
                pd.read_csv(WORKING_DATASET_PATH),
                "Working 工作資料",
                WORKING_DATASET_PATH,
                None,
            )
        fallback = load_dataset()
        return (
            fallback,
            "Original 原始資料",
            ORIGINAL_DATASET_PATH if fallback is not None else None,
            "尚未找到 Working 工作資料，已改用 Original 原始資料。請回到「資料上傳與預覽」重新上傳或建立工作資料。",
        )

    dataset = load_dataset()
    return (
        dataset,
        "Original 原始資料",
        ORIGINAL_DATASET_PATH if dataset is not None else None,
        None,
    )


def _ordered_grouped_frame(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregation: Aggregation,
    *,
    top_n: int,
) -> pd.DataFrame:
    working = df[[x_col]].copy()
    working[x_col] = working[x_col].fillna("Unknown").astype(str)

    if y_col == COUNT_ROWS or aggregation == "count":
        grouped = working.groupby(x_col, dropna=False).size().reset_index(name="value")
    else:
        values = pd.to_numeric(df[y_col], errors="coerce")
        working["value"] = values
        grouped = (
            working.dropna(subset=["value"])
            .groupby(x_col, dropna=False)["value"]
            .agg(aggregation)
            .reset_index(name="value")
        )

    grouped = grouped.sort_values("value", ascending=False).head(top_n)
    return grouped


def _stacked_frame(
    df: pd.DataFrame,
    x_col: str,
    stack_col: str,
    y_col: str,
    aggregation: Aggregation,
    *,
    top_n: int,
) -> pd.DataFrame:
    working = df[[x_col, stack_col]].copy()
    working[x_col] = working[x_col].fillna("Unknown").astype(str)
    working[stack_col] = working[stack_col].fillna("Unknown").astype(str)

    if y_col == COUNT_ROWS or aggregation == "count":
        grouped = working.groupby([x_col, stack_col], dropna=False).size()
    else:
        working["value"] = pd.to_numeric(df[y_col], errors="coerce")
        grouped = (
            working.dropna(subset=["value"])
            .groupby([x_col, stack_col], dropna=False)["value"]
            .agg(aggregation)
        )

    pivot = grouped.unstack(fill_value=0)
    top_index = pivot.sum(axis=1).sort_values(ascending=False).head(top_n).index
    return pivot.loc[top_index]


def _sort_for_line(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    parsed = pd.to_datetime(df[x_col], errors="coerce")
    valid_ratio = parsed.notna().sum() / max(len(df), 1)
    if valid_ratio >= 0.8:
        sorted_df = df.assign(_x_sort=parsed).sort_values("_x_sort")
        return sorted_df.drop(columns=["_x_sort"])

    numeric = pd.to_numeric(df[x_col], errors="coerce")
    numeric_ratio = numeric.notna().sum() / max(len(df), 1)
    if numeric_ratio >= 0.8:
        sorted_df = df.assign(_x_sort=numeric).sort_values("_x_sort")
        return sorted_df.drop(columns=["_x_sort"])

    return df


def _line_frame(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    aggregation: Aggregation,
    group_col: str | None,
) -> pd.DataFrame:
    working = df[[x_col]].copy()
    working[x_col] = working[x_col].fillna("Unknown")

    if group_col:
        working[group_col] = df[group_col].fillna("Unknown").astype(str)

    if y_col == COUNT_ROWS or aggregation == "count":
        group_keys = [x_col] + ([group_col] if group_col else [])
        grouped = working.groupby(group_keys, dropna=False).size().reset_index(name="value")
    else:
        working["value"] = pd.to_numeric(df[y_col], errors="coerce")
        group_keys = [x_col] + ([group_col] if group_col else [])
        grouped = (
            working.dropna(subset=["value"])
            .groupby(group_keys, dropna=False)["value"]
            .agg(aggregation)
            .reset_index(name="value")
        )

    return _sort_for_line(grouped, x_col)


def _new_figure() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(9, 5.2), constrained_layout=True)


def _render_bar_chart(frame: pd.DataFrame, x_col: str, title: str) -> plt.Figure:
    fig, ax = _new_figure()
    ax.bar(frame[x_col].astype(str), frame["value"])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("數值")
    ax.tick_params(axis="x", rotation=35)
    return fig


def _render_pie_chart(frame: pd.DataFrame, x_col: str, title: str) -> plt.Figure | None:
    pie_values = frame[frame["value"] > 0]
    if pie_values.empty:
        return None
    fig, ax = _new_figure()
    ax.pie(
        pie_values["value"],
        labels=pie_values[x_col].astype(str),
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(title)
    ax.axis("equal")
    return fig


def _render_stacked_bar_chart(frame: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = _new_figure()
    frame.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(frame.index.name or "類別")
    ax.set_ylabel("數值")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title=frame.columns.name, bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def _render_line_chart(
    frame: pd.DataFrame,
    x_col: str,
    group_col: str | None,
    title: str,
) -> plt.Figure:
    fig, ax = _new_figure()
    if group_col:
        for name, group in frame.groupby(group_col, dropna=False):
            ax.plot(group[x_col].astype(str), group["value"], marker="o", label=str(name))
        ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.plot(frame[x_col].astype(str), frame["value"], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel("數值")
    ax.tick_params(axis="x", rotation=35)
    return fig


def _render_histogram(df: pd.DataFrame, column: str, bins: int, title: str) -> plt.Figure:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    fig, ax = _new_figure()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("筆數")
    return fig


def _render_radar_chart(
    df: pd.DataFrame,
    columns: list[str],
    aggregation: Literal["mean", "sum", "median"],
    *,
    normalize: bool,
    title: str,
) -> plt.Figure:
    values = []
    labels = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        labels.append(column)
        if aggregation == "sum":
            values.append(float(series.sum()))
        elif aggregation == "median":
            values.append(float(series.median()))
        else:
            values.append(float(series.mean()))

    if normalize and values:
        max_value = max(abs(value) for value in values)
        if max_value:
            values = [value / max_value for value in values]

    angles = [index / float(len(labels)) * 2 * math.pi for index in range(len(labels))]
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 6), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    return fig


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def _chart_filename(chart_type: ChartType) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chart_{chart_type}_{stamp}.png"


def _format_measure(y_col: str, aggregation: str) -> str:
    if y_col == COUNT_ROWS or aggregation == "count":
        return "筆數"
    labels = {"sum": "總和", "mean": "平均", "median": "中位數"}
    return f"{y_col} 的{labels.get(aggregation, aggregation)}"


def _aggregation_value(label: str) -> Aggregation:
    values: dict[str, Aggregation] = {
        "筆數": "count",
        "總和": "sum",
        "平均": "mean",
        "中位數": "median",
    }
    return values.get(label, "count")


def _show_summary(items: dict[str, str | int | None]) -> None:
    st.markdown("##### 圖表設定摘要")
    summary = "  \n".join(
        f"- {key}：{value}" for key, value in items.items() if value not in (None, "")
    )
    st.markdown(summary)


def _suggested_questions(chart_name: str) -> list[str]:
    common = "請根據目前圖表，指出一個值得進一步驗證的觀察。"
    questions = {
        "長條圖": ["哪個類別最高或最低？這個差異可能代表什麼？", common],
        "圓餅圖": ["這個比例分布是否過度集中在少數類別？", common],
        "帶形圖": ["不同堆疊分組的比例差異在哪裡最明顯？", common],
        "折線圖": ["這條線是否有明顯趨勢、轉折或異常點？", common],
        "雷達圖": ["哪些指標特別高或低？是否需要標準化後再比較？", common],
        "直方圖": ["這個分布是否偏斜？有沒有可能的離群值？", common],
    }
    return questions.get(chart_name, [common])


def _warn_if_empty(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("目前資料來源沒有可繪圖資料。請回到「資料上傳與預覽」檢查資料。")
        return True
    return False


main, side = st.columns([5, 3], gap="large")

with main:
    st.title("通用圖表")
    st.caption("自由選擇資料來源、圖表類型與欄位搭配，再下載目前製作出的圖。")

    requested_source = st.radio(
        "資料來源",
        ["Working 工作資料", "Ready 分析就緒資料", "Original 原始資料"],
        horizontal=True,
        key="chart_data_source",
    )
    df, source_label, source_path, source_warning = _load_chart_dataset(requested_source)
    if source_warning:
        st.warning(source_warning)

    if df is None:
        st.info("請先到「資料上傳與預覽」頁上傳 CSV，或先建立 Ready 分析就緒資料。")
    else:
        render_dataset_metrics(df)
        st.caption(f"目前使用：{source_label}")
        if source_path:
            with st.expander("技術資訊", expanded=False):
                st.caption(f"資料檔：`{_display_path(source_path)}`")

        if not _warn_if_empty(df):
            all_cols = _all_columns(df)
            numeric_cols = _numeric_columns(df)
            measures = [COUNT_ROWS] + numeric_cols
            chart_label = st.selectbox(
                "圖表類型",
                ["長條圖", "圓餅圖", "帶形圖", "折線圖", "雷達圖", "直方圖"],
            )
            chart_map: dict[str, ChartType] = {
                "長條圖": "bar",
                "圓餅圖": "pie",
                "帶形圖": "stacked_bar",
                "折線圖": "line",
                "雷達圖": "radar",
                "直方圖": "histogram",
            }
            chart_type = chart_map[chart_label]

            fig: plt.Figure | None = None
            summary: dict[str, str | int | None] = {
                "資料來源": source_label,
                "圖表": chart_label,
            }

            with st.container(border=True):
                st.markdown("##### 圖表設定")

                if chart_type in {"bar", "pie"}:
                    c1, c2, c3 = st.columns(3)
                    x_col = c1.selectbox("分類欄位", all_cols, key=f"{chart_type}_x")
                    y_col = c2.selectbox("數值欄位", measures, key=f"{chart_type}_y")
                    aggregation = c3.selectbox(
                        "聚合方式",
                        ["筆數", "總和", "平均", "中位數"],
                        key=f"{chart_type}_agg",
                    )
                    top_n = st.slider("顯示前 N 類", 3, 30, 10, key=f"{chart_type}_top_n")

                    aggregation_value = _aggregation_value(aggregation)
                    if y_col != COUNT_ROWS and aggregation_value != "count" and y_col not in numeric_cols:
                        st.warning("Y 欄位需要是數值欄，或改用資料筆數。")
                    else:
                        frame = _ordered_grouped_frame(
                            df,
                            x_col,
                            y_col,
                            aggregation_value,  # type: ignore[arg-type]
                            top_n=top_n,
                        )
                        title = f"{x_col} 的 {_format_measure(y_col, aggregation_value)}"
                        if chart_type == "bar":
                            fig = _render_bar_chart(frame, x_col, title)
                        else:
                            fig = _render_pie_chart(frame, x_col, title)
                            if fig is None:
                                st.warning("圓餅圖需要大於 0 的數值，請調整欄位或聚合方式。")
                        summary.update(
                            {
                                "分類欄位": x_col,
                                "數值欄位": y_col,
                                "聚合方式": aggregation,
                                "顯示前 N 類": top_n,
                            }
                        )

                elif chart_type == "stacked_bar":
                    c1, c2 = st.columns(2)
                    x_col = c1.selectbox("X 分類欄位", all_cols, key="stacked_x")
                    stack_col = c2.selectbox("堆疊分組欄位", all_cols, key="stacked_group")
                    c3, c4, c5 = st.columns(3)
                    y_col = c3.selectbox("數值欄位", measures, key="stacked_y")
                    aggregation = c4.selectbox(
                        "聚合方式",
                        ["筆數", "總和", "平均", "中位數"],
                        key="stacked_agg",
                    )
                    top_n = c5.slider("顯示前 N 類", 3, 30, 10, key="stacked_top_n")
                    if x_col == stack_col:
                        st.warning("X 分類欄位和堆疊分組欄位不能相同。")
                    else:
                        aggregation_value = _aggregation_value(aggregation)
                        frame = _stacked_frame(
                            df,
                            x_col,
                            stack_col,
                            y_col,
                            aggregation_value,  # type: ignore[arg-type]
                            top_n=top_n,
                        )
                        title = f"{x_col} 依 {stack_col} 堆疊的 {_format_measure(y_col, aggregation_value)}"
                        fig = _render_stacked_bar_chart(frame, title)
                        summary.update(
                            {
                                "X 分類欄位": x_col,
                                "堆疊分組欄位": stack_col,
                                "數值欄位": y_col,
                                "聚合方式": aggregation,
                                "顯示前 N 類": top_n,
                            }
                        )

                elif chart_type == "line":
                    c1, c2 = st.columns(2)
                    x_col = c1.selectbox("X 軸欄位", all_cols, key="line_x")
                    y_col = c2.selectbox("Y 軸欄位", measures, key="line_y")
                    c3, c4 = st.columns(2)
                    aggregation = c3.selectbox(
                        "聚合方式",
                        ["筆數", "總和", "平均", "中位數"],
                        key="line_agg",
                    )
                    group_options = ["不分組"] + all_cols
                    group_choice = c4.selectbox("分組欄位", group_options, key="line_group")
                    group_col = None if group_choice == "不分組" else group_choice
                    if group_col == x_col:
                        st.warning("X 軸欄位和分組欄位不能相同。")
                    else:
                        aggregation_value = _aggregation_value(aggregation)
                        frame = _line_frame(
                            df,
                            x_col,
                            y_col,
                            aggregation_value,  # type: ignore[arg-type]
                            group_col,
                        )
                        title = f"{_format_measure(y_col, aggregation_value)} 隨 {x_col} 變化"
                        fig = _render_line_chart(frame, x_col, group_col, title)
                        summary.update(
                            {
                                "X 軸欄位": x_col,
                                "Y 軸欄位": y_col,
                                "聚合方式": aggregation,
                                "分組欄位": group_col,
                            }
                        )

                elif chart_type == "histogram":
                    if not numeric_cols:
                        st.warning("直方圖需要至少一個數值欄位。")
                    else:
                        c1, c2 = st.columns(2)
                        x_col = c1.selectbox("數值欄位", numeric_cols, key="hist_x")
                        bins = c2.slider("分箱數", 5, 80, 20, key="hist_bins")
                        title = f"{x_col} 分布"
                        fig = _render_histogram(df, x_col, bins, title)
                        summary.update({"數值欄位": x_col, "分箱數": bins})

                elif chart_type == "radar":
                    if len(numeric_cols) < 3:
                        st.warning("雷達圖建議至少選 3 個數值欄位。")
                    else:
                        selected = st.multiselect(
                            "數值欄位",
                            numeric_cols,
                            default=numeric_cols[: min(5, len(numeric_cols))],
                            key="radar_cols",
                        )
                        c1, c2 = st.columns(2)
                        aggregation = c1.selectbox(
                            "聚合方式",
                            ["平均", "總和", "中位數"],
                            key="radar_agg",
                        )
                        normalize = c2.checkbox("標準化到相同尺度", value=True)
                        if len(selected) < 3:
                            st.warning("雷達圖至少需要 3 個數值欄位。")
                        else:
                            aggregation_value = _aggregation_value(aggregation)
                            title = f"雷達圖（{aggregation}）"
                            fig = _render_radar_chart(
                                df,
                                selected,
                                aggregation_value,  # type: ignore[arg-type]
                                normalize=normalize,
                                title=title,
                            )
                            summary.update(
                                {
                                    "數值欄位": ", ".join(selected),
                                    "聚合方式": aggregation,
                                    "標準化": "是" if normalize else "否",
                                }
                            )

            if fig is not None:
                st.markdown("##### 預覽")
                st.pyplot(fig, clear_figure=False)
                st.download_button(
                    "下載目前圖表 PNG",
                    data=_figure_to_png_bytes(fig),
                    file_name=_chart_filename(chart_type),
                    mime="image/png",
                )
                _show_summary(summary)
                plt.close(fig)

                st.markdown("##### 建議問 Agent")
                st.markdown(
                    "\n".join(f"- {question}" for question in _suggested_questions(chart_label))
                )

with side:
    render_chat_panel(page_name="通用圖表")
