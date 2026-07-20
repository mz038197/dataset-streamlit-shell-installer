from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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


st.set_page_config(page_title="圖表探索", page_icon="CH", layout="wide")
inject_style()

COUNT_ROWS = "資料筆數"
PLEASE_SELECT = "請選擇"
ChartType = Literal["bar", "pie", "stacked_bar", "line", "radar", "histogram"]
Aggregation = Literal["count", "sum", "mean", "median"]
configure_matplotlib_for_traditional_chinese()

CHART_LABELS: list[str] = ["長條圖", "圓餅圖", "帶形圖", "折線圖", "雷達圖", "直方圖"]
CHART_LABEL_TO_KEY: dict[str, ChartType] = {
    "長條圖": "bar",
    "圓餅圖": "pie",
    "帶形圖": "stacked_bar",
    "折線圖": "line",
    "雷達圖": "radar",
    "直方圖": "histogram",
}
CHART_KEY_TO_LABEL: dict[ChartType, str] = {value: key for key, value in CHART_LABEL_TO_KEY.items()}


@dataclass(frozen=True)
class QuizItem:
    id: str
    prompt: str
    correct: ChartType
    agent_hints: tuple[str, ...]


QUIZ_ITEMS: tuple[QuizItem, ...] = (
    QuizItem(
        "compare_category",
        "比較類別數量",
        "bar",
        ("這一題在比較不同類別的多寡，適合哪一種圖？", "為什麼這一題比較不適合用圓餅圖？"),
    ),
    QuizItem(
        "overall_proportion",
        "整體比例（非時間）",
        "pie",
        ("整體占比要用哪一種圖比較清楚？", "類別很多時，圓餅圖可能有什麼問題？"),
    ),
    QuizItem(
        "stacked_composition",
        "多組比例構成",
        "stacked_bar",
        ("想同時看「分組」又看「裡面怎麼組成」，該選哪種圖？", "帶形圖和一般長條圖差在哪裡？"),
    ),
    QuizItem(
        "trend",
        "時間或連續趨勢",
        "line",
        ("時間或連續變化該選哪種圖？", "為什麼這一題不該用圓餅圖？"),
    ),
    QuizItem(
        "radar_balance",
        "多指標平衡比較",
        "radar",
        ("要同時看多個數值指標是否平衡，適合哪種圖？", "雷達圖為什麼常需要標準化？"),
    ),
    QuizItem(
        "distribution",
        "定量資料區間分布",
        "histogram",
        ("想看某個數值落在哪些區間，該選哪種圖？", "直方圖和長條圖差在哪裡？"),
    ),
)


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
            None
            if dataset is not None
            else "尚未建立 Ready 分析就緒資料，請先到「建立 Ready 分析就緒資料」頁產生 ready.csv。",
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


def _is_near_unique(series: pd.Series, n_rows: int) -> bool:
    if n_rows <= 0:
        return False
    nunique = int(series.nunique(dropna=True))
    return nunique >= max(20, int(n_rows * 0.9))


def _datetime_ratio(series: pd.Series) -> float:
    """Only meaningful for non-numeric text-like columns (numbers parse as dates too easily)."""
    if series.empty or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return 0.0
    sample = series.dropna()
    if sample.empty:
        return 0.0
    if len(sample) > 200:
        sample = sample.sample(200, random_state=0)
    with pd.option_context("mode.chained_assignment", None):
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().sum()) / float(len(sample))


def _infer_column_roles(df: pd.DataFrame) -> dict[str, list[str]]:
    """Type/cardinality roles only — no column-name dictionaries.

    All numeric columns stay chartable (measures). High-cardinality text is
    skipped as free-text. Defaults use 資料筆數 or statistical preference.
    """
    n_rows = len(df)
    categorical: list[str] = []
    measures: list[str] = []
    datetime_cols: list[str] = []
    continuous: list[str] = []

    for column in df.columns:
        name = str(column)
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(name)
            continue

        if pd.api.types.is_bool_dtype(series):
            categorical.append(name)
            continue

        if pd.api.types.is_numeric_dtype(series):
            nunique = int(series.nunique(dropna=True))
            low_card_cap = min(20, max(3, int(n_rows * 0.05)) if n_rows else 20)
            measures.append(name)
            if pd.api.types.is_integer_dtype(series) and 1 < nunique <= low_card_cap:
                # Usable as category (bar/pie/stacked) and still a measure (radar).
                categorical.append(name)
            else:
                continuous.append(name)
            continue

        # object / string / category — datetime before treating as category
        if _datetime_ratio(series) >= 0.8:
            datetime_cols.append(name)
            continue

        # High-cardinality text → free text / labels, skip as chart category
        if _is_near_unique(series, n_rows):
            continue

        nunique = int(series.nunique(dropna=True))
        if 1 < nunique <= min(50, max(3, n_rows)):
            categorical.append(name)

    categorical = sorted(
        categorical,
        key=lambda col: (int(df[col].nunique(dropna=True)), col),
    )
    return {
        "categorical": categorical,
        "measures": measures,
        "continuous": continuous,
        "datetime": datetime_cols,
    }


def _looks_like_surrogate_key(series: pd.Series) -> bool:
    """Near-unique ints aligned with row order (0/1-based) — default-demote only."""
    n = len(series)
    if n <= 0:
        return False
    values = pd.to_numeric(series, errors="coerce")
    if not _is_near_unique(values, n):
        return False
    filled = values.dropna()
    if len(filled) < 2:
        return False
    if not (filled == filled.round()).all():
        return False
    positions_1 = pd.Series(range(1, n + 1), index=series.index, dtype="float64")
    positions_0 = pd.Series(range(n), index=series.index, dtype="float64")
    match_1 = float((values == positions_1).sum()) / float(n)
    match_0 = float((values == positions_0).sum()) / float(n)
    return max(match_1, match_0) >= 0.9


def _prefer_numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Rank numerics for picker order — surrogate keys last, never removed."""

    def score(col: str) -> tuple[float, float, float, float, str]:
        series = df[col]
        values = pd.to_numeric(series, errors="coerce")
        n = max(len(df), 1)
        nunique = int(values.nunique(dropna=True))
        has_dup = nunique < max(2, int(n * 0.95))
        is_float = bool(pd.api.types.is_float_dtype(series))
        if not is_float and values.notna().any():
            sample = values.dropna()
            is_float = not bool((sample == sample.round()).all())
        variance = float(values.var(skipna=True)) if values.notna().sum() > 1 else 0.0
        if pd.isna(variance):
            variance = 0.0
        return (
            0.0 if _looks_like_surrogate_key(series) else 1.0,
            1.0 if is_float else 0.0,
            1.0 if has_dup else 0.0,
            variance,
            col,
        )

    return sorted(columns, key=score, reverse=True)


def _safe_numeric_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Ranked numerics with surrogate keys removed — for auto-defaults only."""
    return [
        col
        for col in _prefer_numeric_columns(df, columns)
        if not _looks_like_surrogate_key(df[col])
    ]


def _group_sizes_mostly_one(df: pd.DataFrame, x_col: str) -> bool:
    """True when most X groups have a single row — count(Y) would be a flat line."""
    if x_col not in df.columns or df.empty:
        return True
    sizes = df.groupby(df[x_col], dropna=False).size()
    if sizes.empty:
        return True
    return float((sizes <= 1).mean()) >= 0.8


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
) -> plt.Figure | None:
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

    if len(labels) < 3:
        return None

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


def _aggregation_label(value: str) -> str:
    labels = {
        "count": "筆數",
        "sum": "總和",
        "mean": "平均",
        "median": "中位數",
    }
    return labels.get(value, "筆數")


def _show_summary(items: dict[str, str | int | None]) -> None:
    st.markdown("##### 圖表設定摘要")
    summary = "  \n".join(
        f"- {key}：{value}" for key, value in items.items() if value not in (None, "")
    )
    st.markdown(summary)


def _warn_if_empty(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("目前資料來源沒有可繪圖資料。請回到「資料上傳與預覽」檢查資料。")
        return True
    return False


def _quiz_choice_key(item_id: str) -> str:
    return f"chart_quiz_choice_{item_id}"


def _quiz_drawn_key(item_id: str) -> str:
    return f"chart_quiz_drawn_{item_id}"


def _selection_is_correct(item: QuizItem, choice_label: str) -> bool:
    if choice_label == PLEASE_SELECT:
        return False
    return CHART_LABEL_TO_KEY.get(choice_label) == item.correct


def _build_agent_context(
    *,
    source_label: str,
    focus_id: str | None,
    drawn_id: str | None,
    draw_summary: dict[str, str | int | None] | None,
) -> str:
    lines = [
        "目前頁面：圖表探索（先選對圖種，再用自己的資料畫圖）。",
        "選圖＝先問清楚要比較什麼（類別／比例／時間／關係）。",
        f"資料來源：{source_label}。",
        "請協助學生選出正確圖種、解釋為何選錯、並在畫圖後解讀圖與數據表；不要在對話中重畫圖。",
        "不要直接洩漏尚未作答題目的標準答案，除非學生明確要求揭曉；可給思考方向。",
        "",
        "六題作答狀態：",
    ]
    preview_rendered = bool(draw_summary) and "狀態" not in draw_summary
    for index, item in enumerate(QUIZ_ITEMS, start=1):
        choice = st.session_state.get(_quiz_choice_key(item.id), PLEASE_SELECT)
        correct = _selection_is_correct(item, choice)
        drawn = bool(st.session_state.get(_quiz_drawn_key(item.id), False)) and correct
        status = "已解鎖可畫圖" if correct else ("未選" if choice == PLEASE_SELECT else "與建議不符")
        focus_mark = "（目前焦點）" if item.id == focus_id else ""
        drawn_mark = (
            "；已畫出預覽"
            if drawn and item.id == drawn_id and preview_rendered
            else ""
        )
        lines.append(
            f"{index}. 題幹「{item.prompt}」{focus_mark}：學生選擇「{choice}」→ {status}{drawn_mark}"
        )

    if draw_summary:
        lines.append("")
        lines.append("目前預覽圖摘要：")
        for key, value in draw_summary.items():
            if value not in (None, ""):
                lines.append(f"- {key}：{value}")
    return "\n".join(lines)


def _quiz_agent_hints(focus_id: str | None) -> list[str]:
    item = next((entry for entry in QUIZ_ITEMS if entry.id == focus_id), None)
    if item is None:
        return [
            "這一題為什麼不能選圓餅圖？",
            "時間或連續趨勢該選哪種圖？為什麼？",
            "選圖前要先問清楚要比較什麼（類別／比例／時間／關係）？",
        ]
    return list(item.agent_hints) + ["請根據題幹，說明選圖時該先問清楚什麼。"]


def _default_draw_config(
    chart_type: ChartType,
    roles: dict[str, list[str]],
    df: pd.DataFrame,
) -> tuple[dict[str, Any] | None, str | None]:
    cats = roles["categorical"]
    measures = roles["measures"]
    continuous = roles["continuous"]
    datetimes = roles["datetime"]
    # Auto-defaults exclude surrogate keys; pickers still use full measures/continuous.
    safe_measures = _safe_numeric_columns(df, measures)
    safe_continuous = _safe_numeric_columns(df, continuous)

    if chart_type in {"bar", "pie"}:
        if not cats:
            return None, "這份資料目前缺少適合的類別欄，還不能畫這張圖。請換意圖或先整理資料。"
        return {"x_col": cats[0], "y_col": COUNT_ROWS, "aggregation": "count", "top_n": 10}, None

    if chart_type == "stacked_bar":
        if len(cats) < 2:
            return None, "帶形圖需要至少兩個類別欄（X 與堆疊分組）。這份資料目前不足。"
        return {
            "x_col": cats[0],
            "stack_col": cats[1],
            "y_col": COUNT_ROWS,
            "aggregation": "count",
            "top_n": 10,
        }, None

    if chart_type == "line":
        x_candidates = list(datetimes) + safe_continuous + safe_measures
        x_candidates = list(dict.fromkeys(x_candidates))
        x_col = x_candidates[0] if x_candidates else None
        if x_col is None:
            return None, (
                "折線圖需要日期欄或連續數值欄當 X（已排除列號／ID）。"
                "目前無法自動預選，請換有真正量測欄的資料。"
            )
        y_col: str = COUNT_ROWS
        aggregation: Aggregation = "count"
        # Unique dates make count(Y) a flat line of 1s — need a real measure, not an ID.
        if _group_sizes_mostly_one(df, x_col):
            y_candidates = [col for col in safe_measures if col != x_col]
            if not y_candidates:
                return None, (
                    "折線圖在此資料上需要可聚合的數值欄（已排除列號／ID）。"
                    "目前無法自動預選，請勿以 ID 湊數。"
                )
            y_col = y_candidates[0]
            aggregation = "mean"
        note = None
        if not datetimes:
            note = "這份資料沒有時間欄，改看連續趨勢。"
        return {
            "x_col": x_col,
            "y_col": y_col,
            "aggregation": aggregation,
            "group_col": None,
            "no_datetime_note": note,
        }, None

    if chart_type == "radar":
        if len(safe_measures) < 3:
            return None, (
                "雷達圖需要至少 3 個數值欄（已排除列號／ID）。"
                "目前無法自動預選，請勿以 ID 湊數。"
            )
        return {
            "columns": safe_measures[: min(5, len(safe_measures))],
            "aggregation": "mean",
            "normalize": True,
        }, None

    if chart_type == "histogram":
        hist_candidates = safe_continuous or safe_measures
        if not hist_candidates:
            return None, (
                "直方圖需要至少一個數值欄（已排除列號／ID）。"
                "目前無法自動預選，請勿以 ID 湊數。"
            )
        return {"column": hist_candidates[0], "bins": 20}, None

    return None, "未知圖種。"


def _manual_selection_config(
    chart_type: ChartType,
    roles: dict[str, list[str]],
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Build a non-drawing config when safe defaults are unavailable.

    Full numeric choices remain available in the widgets, but a surrogate key
    is never selected automatically.  The placeholder values are checked
    before rendering.
    """
    numeric_cols = roles["measures"]
    if not numeric_cols:
        return None

    safe_measures = _safe_numeric_columns(df, numeric_cols)
    safe_continuous = _safe_numeric_columns(df, roles["continuous"])

    if chart_type == "line":
        safe_x = list(dict.fromkeys(roles["datetime"] + safe_continuous + safe_measures))
        return {
            "x_col": safe_x[0] if safe_x else PLEASE_SELECT,
            "y_col": PLEASE_SELECT,
            "aggregation": "mean",
            "group_col": None,
            "no_datetime_note": None
            if roles["datetime"]
            else "這份資料沒有時間欄，改看連續趨勢。",
        }

    if chart_type == "histogram":
        return {"column": PLEASE_SELECT, "bins": 20}

    if chart_type == "radar" and len(numeric_cols) >= 3:
        return {
            "columns": safe_measures[: min(5, len(safe_measures))],
            "aggregation": "mean",
            "normalize": True,
        }

    return None


def _render_guided_chart(
    df: pd.DataFrame,
    chart_type: ChartType,
    config: dict[str, Any],
) -> tuple[plt.Figure | None, pd.DataFrame | None, dict[str, str | int | None], str | None]:
    summary: dict[str, str | int | None] = {"圖表": CHART_KEY_TO_LABEL[chart_type]}
    table: pd.DataFrame | None = None

    if chart_type in {"bar", "pie"}:
        x_col = str(config["x_col"])
        y_col = str(config["y_col"])
        aggregation: Aggregation = config["aggregation"]  # type: ignore[assignment]
        top_n = int(config["top_n"])
        frame = _ordered_grouped_frame(df, x_col, y_col, aggregation, top_n=top_n)
        title = f"{x_col} 的 {_format_measure(y_col, aggregation)}"
        fig = (
            _render_bar_chart(frame, x_col, title)
            if chart_type == "bar"
            else _render_pie_chart(frame, x_col, title)
        )
        if fig is None:
            return None, None, summary, "圓餅圖需要大於 0 的數值，請調整欄位或聚合方式。"
        table = frame.rename(columns={"value": "數值"})
        summary.update(
            {
                "分類欄位": x_col,
                "數值欄位": y_col,
                "聚合方式": aggregation,
                "顯示前 N 類": top_n,
            }
        )
        return fig, table, summary, None

    if chart_type == "stacked_bar":
        x_col = str(config["x_col"])
        stack_col = str(config["stack_col"])
        y_col = str(config["y_col"])
        aggregation = config["aggregation"]  # type: ignore[assignment]
        top_n = int(config["top_n"])
        if x_col == stack_col:
            return None, None, summary, "X 分類欄位和堆疊分組欄位不能相同。"
        frame = _stacked_frame(df, x_col, stack_col, y_col, aggregation, top_n=top_n)
        title = f"{x_col} 依 {stack_col} 堆疊的 {_format_measure(y_col, aggregation)}"
        fig = _render_stacked_bar_chart(frame, title)
        table = frame.reset_index()
        summary.update(
            {
                "X 分類欄位": x_col,
                "堆疊分組欄位": stack_col,
                "數值欄位": y_col,
                "聚合方式": aggregation,
                "顯示前 N 類": top_n,
            }
        )
        return fig, table, summary, None

    if chart_type == "line":
        x_col = str(config["x_col"])
        y_col = str(config["y_col"])
        aggregation = config["aggregation"]  # type: ignore[assignment]
        group_col = config.get("group_col")
        frame = _line_frame(df, x_col, y_col, aggregation, group_col)
        title = f"{_format_measure(y_col, aggregation)} 隨 {x_col} 變化"
        fig = _render_line_chart(frame, x_col, group_col, title)
        table = frame.rename(columns={"value": "數值"})
        summary.update(
            {
                "X 軸欄位": x_col,
                "Y 軸欄位": y_col,
                "聚合方式": aggregation,
                "分組欄位": group_col or "不分組",
            }
        )
        note = config.get("no_datetime_note")
        return fig, table, summary, note if isinstance(note, str) else None

    if chart_type == "radar":
        columns = list(config["columns"])
        aggregation_label = str(config.get("aggregation_label", "平均"))
        aggregation_value = _aggregation_value(aggregation_label)
        if aggregation_value == "count":
            aggregation_value = "mean"
        normalize = bool(config.get("normalize", True))
        fig = _render_radar_chart(
            df,
            columns,
            aggregation_value,  # type: ignore[arg-type]
            normalize=normalize,
            title=f"雷達圖（{aggregation_label}）",
        )
        if fig is None:
            return None, None, summary, "雷達圖至少需要 3 個有效數值欄位。"
        rows = []
        for column in columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                continue
            if aggregation_value == "sum":
                value = float(series.sum())
            elif aggregation_value == "median":
                value = float(series.median())
            else:
                value = float(series.mean())
            rows.append({"指標": column, "數值": value})
        table = pd.DataFrame(rows)
        summary.update(
            {
                "數值欄位": ", ".join(columns),
                "聚合方式": aggregation_label,
                "標準化": "是" if normalize else "否",
            }
        )
        return fig, table, summary, None

    if chart_type == "histogram":
        column = str(config["column"])
        bins = int(config["bins"])
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        fig = _render_histogram(df, column, bins, f"{column} 分布")
        binned = pd.cut(values, bins=bins)
        table = (
            binned.value_counts()
            .sort_index()
            .rename_axis("區間")
            .reset_index(name="筆數")
        )
        table["區間"] = table["區間"].astype(str)
        summary.update({"數值欄位": column, "分箱數": bins})
        return fig, table, summary, None

    return None, None, summary, "未知圖種。"


def _render_misuse_contrast(df: pd.DataFrame, roles: dict[str, list[str]]) -> None:
    datetimes = roles["datetime"]
    continuous = roles["continuous"] or roles["measures"]
    st.markdown("##### 常見誤用對照")
    st.caption("對齊講義：圓餅圖不適合呈現時間或連續變化；也别只看圖不看數。")

    if datetimes:
        x_col = datetimes[0]
        frame = _ordered_grouped_frame(df, x_col, COUNT_ROWS, "count", top_n=12)
        bad = _render_pie_chart(frame, x_col, f"❌ 誤用：把「{x_col}」當成圓餅比例")
        good_frame = _line_frame(df, x_col, COUNT_ROWS, "count", None)
        good = _render_line_chart(good_frame, x_col, None, f"✅ 建議：用折線看「{x_col}」趨勢")
        col_a, col_b = st.columns(2)
        with col_a:
            if bad is not None:
                st.pyplot(bad, clear_figure=True)
                plt.close(bad)
            else:
                st.info("無法產生誤用圓餅示例。")
        with col_b:
            st.pyplot(good, clear_figure=True)
            plt.close(good)
        return

    if continuous:
        column = continuous[0]
        values = pd.to_numeric(df[column], errors="coerce")
        binned = pd.cut(values, bins=8)
        frame = (
            binned.astype(str)
            .fillna("Unknown")
            .value_counts()
            .rename_axis(column)
            .reset_index(name="value")
        )
        bad = _render_pie_chart(frame, column, f"❌ 誤用：把連續「{column}」切段當圓餅")
        good = _render_histogram(df, column, 20, f"✅ 建議：用直方圖看「{column}」分布")
        col_a, col_b = st.columns(2)
        with col_a:
            if bad is not None:
                st.pyplot(bad, clear_figure=True)
                plt.close(bad)
            else:
                st.info("無法產生誤用圓餅示例。")
        with col_b:
            st.pyplot(good, clear_figure=True)
            plt.close(good)
        return

    st.info("目前資料沒有日期或連續數值欄，略過誤用對照示例。")


def _render_quiz_preview(
    df: pd.DataFrame,
    item: QuizItem,
    roles: dict[str, list[str]],
    source_label: str,
) -> dict[str, str | int | None] | None:
    chart_type = item.correct
    defaults, missing = _default_draw_config(chart_type, roles, df)
    st.markdown(f"##### 預覽：{item.prompt} → {CHART_KEY_TO_LABEL[chart_type]}")
    manual_selection_required = False
    if missing:
        manual_config = _manual_selection_config(chart_type, roles, df)
        if manual_config is None:
            st.warning(missing)
            return {
                "資料來源": source_label,
                "題幹": item.prompt,
                "圖表": CHART_KEY_TO_LABEL[chart_type],
                "狀態": missing,
            }
        st.info(f"{missing} 你仍可從下方明確選擇欄位後畫圖。")
        defaults = manual_config
        manual_selection_required = True

    assert defaults is not None
    all_cols = _all_columns(df)
    numeric_cols = _numeric_columns(df)
    y_options = [COUNT_ROWS] + numeric_cols
    cats = roles["categorical"] or all_cols
    config = dict(defaults)

    with st.container(border=True):
        st.markdown("###### 欄位微調")
        st.caption("已預選欄位（類別圖預設筆數）；可自行更改。數值 ID 仍可手動選。")
        agg_labels = ["筆數", "總和", "平均", "中位數"]
        if chart_type in {"bar", "pie"}:
            c1, c2, c3, c4 = st.columns(4)
            x_default = config["x_col"] if config["x_col"] in cats else cats[0]
            config["x_col"] = c1.selectbox(
                "分類欄位",
                cats,
                index=cats.index(x_default) if x_default in cats else 0,
                key=f"guided_{item.id}_x",
            )
            y_default = config["y_col"] if config["y_col"] in y_options else y_options[0]
            config["y_col"] = c2.selectbox(
                "數值欄位",
                y_options,
                index=y_options.index(y_default),
                key=f"guided_{item.id}_y",
            )
            agg_default = _aggregation_label(str(config.get("aggregation", "count")))
            config["aggregation"] = _aggregation_value(
                c3.selectbox(
                    "聚合方式",
                    agg_labels,
                    index=agg_labels.index(agg_default),
                    key=f"guided_{item.id}_agg",
                )
            )
            config["top_n"] = c4.slider("顯示前 N 類", 3, 30, int(config["top_n"]), key=f"guided_{item.id}_top")

        elif chart_type == "stacked_bar":
            c1, c2 = st.columns(2)
            x_default = config["x_col"] if config["x_col"] in cats else cats[0]
            stack_default = config["stack_col"] if config["stack_col"] in cats else cats[min(1, len(cats) - 1)]
            config["x_col"] = c1.selectbox(
                "X 分類欄位",
                cats,
                index=cats.index(x_default),
                key=f"guided_{item.id}_x",
            )
            config["stack_col"] = c2.selectbox(
                "堆疊分組欄位",
                cats,
                index=cats.index(stack_default),
                key=f"guided_{item.id}_stack",
            )
            c3, c4, c5 = st.columns(3)
            y_default = config["y_col"] if config["y_col"] in y_options else y_options[0]
            config["y_col"] = c3.selectbox(
                "數值欄位",
                y_options,
                index=y_options.index(y_default),
                key=f"guided_{item.id}_y",
            )
            agg_default = _aggregation_label(str(config.get("aggregation", "count")))
            config["aggregation"] = _aggregation_value(
                c4.selectbox(
                    "聚合方式",
                    agg_labels,
                    index=agg_labels.index(agg_default),
                    key=f"guided_{item.id}_agg",
                )
            )
            config["top_n"] = c5.slider("顯示前 N 類", 3, 30, int(config["top_n"]), key=f"guided_{item.id}_top")

        elif chart_type == "line":
            x_candidates = (
                roles["datetime"]
                + _prefer_numeric_columns(df, roles["continuous"])
                + _prefer_numeric_columns(df, roles["measures"])
            )
            x_candidates = list(dict.fromkeys(x_candidates)) or all_cols
            if manual_selection_required:
                x_candidates = [PLEASE_SELECT] + x_candidates
            c1, c2 = st.columns(2)
            x_default = config["x_col"] if config["x_col"] in x_candidates else x_candidates[0]
            config["x_col"] = c1.selectbox(
                "X 軸欄位",
                x_candidates,
                index=x_candidates.index(x_default),
                key=f"guided_{item.id}_x",
            )
            line_y_options = ([PLEASE_SELECT] + y_options) if manual_selection_required else y_options
            y_default = (
                config["y_col"] if config["y_col"] in line_y_options else line_y_options[0]
            )
            config["y_col"] = c2.selectbox(
                "Y 軸欄位",
                line_y_options,
                index=line_y_options.index(y_default),
                key=f"guided_{item.id}_y",
            )
            agg_default = _aggregation_label(str(config.get("aggregation", "count")))
            config["aggregation"] = _aggregation_value(
                st.selectbox(
                    "聚合方式",
                    agg_labels,
                    index=agg_labels.index(agg_default),
                    key=f"guided_{item.id}_agg",
                )
            )
            if not roles["datetime"]:
                config["no_datetime_note"] = "這份資料沒有時間欄，改看連續趨勢。"

        elif chart_type == "radar":
            default_cols = [col for col in config["columns"] if col in numeric_cols] or numeric_cols[:3]
            selected = st.multiselect(
                "數值欄位",
                numeric_cols,
                default=default_cols[: min(5, len(default_cols))],
                key=f"guided_{item.id}_cols",
            )
            c1, c2 = st.columns(2)
            aggregation_label = c1.selectbox(
                "聚合方式",
                ["平均", "總和", "中位數"],
                key=f"guided_{item.id}_agg",
            )
            normalize = c2.checkbox("標準化到相同尺度", value=True, key=f"guided_{item.id}_norm")
            config = {
                "columns": selected,
                "aggregation_label": aggregation_label,
                "normalize": normalize,
            }

        elif chart_type == "histogram":
            hist_cols = numeric_cols
            if not hist_cols:
                st.warning("直方圖需要至少一個數值欄。")
                return {
                    "資料來源": source_label,
                    "題幹": item.prompt,
                    "圖表": CHART_KEY_TO_LABEL[chart_type],
                    "狀態": "缺少數值欄",
                }
            if manual_selection_required:
                hist_cols = [PLEASE_SELECT] + hist_cols
            c1, c2 = st.columns(2)
            col_default = config["column"] if config["column"] in hist_cols else hist_cols[0]
            config["column"] = c1.selectbox(
                "數值欄位",
                hist_cols,
                index=hist_cols.index(col_default),
                key=f"guided_{item.id}_col",
            )
            config["bins"] = c2.slider("分箱數", 5, 80, int(config["bins"]), key=f"guided_{item.id}_bins")

    waiting_for_selection = (
        (chart_type == "line" and PLEASE_SELECT in {config.get("x_col"), config.get("y_col")})
        or (chart_type == "histogram" and config.get("column") == PLEASE_SELECT)
        or (chart_type == "radar" and len(config.get("columns", [])) < 3)
    )
    if waiting_for_selection:
        st.info("請先完成欄位選擇；系統不會用列號／ID 自動補位。")
        return {
            "資料來源": source_label,
            "題幹": item.prompt,
            "圖表": CHART_KEY_TO_LABEL[chart_type],
            "狀態": "等待選擇欄位",
        }

    fig, table, summary, note = _render_guided_chart(df, chart_type, config)
    summary = {"資料來源": source_label, "題幹": item.prompt, **summary}
    if note:
        st.info(note)
    if fig is None:
        st.warning(note or "無法繪製圖表，請調整欄位。")
        return summary

    st.pyplot(fig, clear_figure=False)
    st.download_button(
        "下載目前圖表 PNG",
        data=_figure_to_png_bytes(fig),
        file_name=_chart_filename(chart_type),
        mime="image/png",
        key=f"guided_download_{item.id}",
    )
    plt.close(fig)
    if table is not None:
        st.markdown("##### 對應數據表")
        st.dataframe(table, use_container_width=True, hide_index=True)
    _show_summary(summary)

    if chart_type in {"line", "histogram"}:
        _render_misuse_contrast(df, roles)

    return summary


def _render_advanced_charts(df: pd.DataFrame, source_label: str) -> None:
    all_cols = _all_columns(df)
    numeric_cols = _numeric_columns(df)
    measures = [COUNT_ROWS] + numeric_cols
    chart_label = st.selectbox(
        "圖表類型",
        CHART_LABELS,
        key="advanced_chart_type",
    )
    chart_type = CHART_LABEL_TO_KEY[chart_label]
    fig: plt.Figure | None = None
    summary: dict[str, str | int | None] = {
        "資料來源": source_label,
        "圖表": chart_label,
    }

    with st.container(border=True):
        st.markdown("##### 圖表設定")

        if chart_type in {"bar", "pie"}:
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("分類欄位", all_cols, key=f"adv_{chart_type}_x")
            y_col = c2.selectbox("數值欄位", measures, key=f"adv_{chart_type}_y")
            aggregation = c3.selectbox(
                "聚合方式",
                ["筆數", "總和", "平均", "中位數"],
                key=f"adv_{chart_type}_agg",
            )
            top_n = st.slider("顯示前 N 類", 3, 30, 10, key=f"adv_{chart_type}_top_n")
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
            x_col = c1.selectbox("X 分類欄位", all_cols, key="adv_stacked_x")
            stack_col = c2.selectbox("堆疊分組欄位", all_cols, key="adv_stacked_group")
            c3, c4, c5 = st.columns(3)
            y_col = c3.selectbox("數值欄位", measures, key="adv_stacked_y")
            aggregation = c4.selectbox(
                "聚合方式",
                ["筆數", "總和", "平均", "中位數"],
                key="adv_stacked_agg",
            )
            top_n = c5.slider("顯示前 N 類", 3, 30, 10, key="adv_stacked_top_n")
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
            x_col = c1.selectbox("X 軸欄位", all_cols, key="adv_line_x")
            y_col = c2.selectbox("Y 軸欄位", measures, key="adv_line_y")
            c3, c4 = st.columns(2)
            aggregation = c3.selectbox(
                "聚合方式",
                ["筆數", "總和", "平均", "中位數"],
                key="adv_line_agg",
            )
            group_options = ["不分組"] + all_cols
            group_choice = c4.selectbox("分組欄位", group_options, key="adv_line_group")
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
                x_col = c1.selectbox("數值欄位", numeric_cols, key="adv_hist_x")
                bins = c2.slider("分箱數", 5, 80, 20, key="adv_hist_bins")
                fig = _render_histogram(df, x_col, bins, f"{x_col} 分布")
                summary.update({"數值欄位": x_col, "分箱數": bins})

        elif chart_type == "radar":
            if len(numeric_cols) < 3:
                st.warning("雷達圖建議至少選 3 個數值欄位。")
            else:
                selected = st.multiselect(
                    "數值欄位",
                    numeric_cols,
                    default=numeric_cols[: min(5, len(numeric_cols))],
                    key="adv_radar_cols",
                )
                c1, c2 = st.columns(2)
                aggregation = c1.selectbox(
                    "聚合方式",
                    ["平均", "總和", "中位數"],
                    key="adv_radar_agg",
                )
                normalize = c2.checkbox("標準化到相同尺度", value=True, key="adv_radar_norm")
                if len(selected) < 3:
                    st.warning("雷達圖至少需要 3 個數值欄位。")
                else:
                    aggregation_value = _aggregation_value(aggregation)
                    fig = _render_radar_chart(
                        df,
                        selected,
                        aggregation_value,  # type: ignore[arg-type]
                        normalize=normalize,
                        title=f"雷達圖（{aggregation}）",
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
            key="adv_download",
        )
        _show_summary(summary)
        plt.close(fig)


def run_page() -> None:
    if "chart_quiz_focus" not in st.session_state:
        st.session_state["chart_quiz_focus"] = QUIZ_ITEMS[0].id

    main, side = st.columns([5, 3], gap="large")
    draw_summary_for_agent: dict[str, str | int | None] | None = None
    source_label_for_agent = "尚未載入"

    with main:
        st.title("圖表探索")
        st.caption(
            "選圖＝先問清楚要比較什麼（類別／比例／時間／關係）。"
            "本頁六題對應講義 CONCEPT 02；先選對圖種，再用你的資料畫圖。可問右側 Agent。"
        )

        requested_source = st.radio(
            "資料來源",
            ["Working 工作資料", "Ready 分析就緒資料", "Original 原始資料"],
            horizontal=True,
            key="chart_data_source",
        )
        df, source_label, source_path, source_warning = _load_chart_dataset(requested_source)
        source_label_for_agent = source_label
        if source_warning:
            st.warning(source_warning)

        if df is None:
            st.info("請先到「資料上傳與預覽」頁上傳 CSV，或先建立 Ready 分析就緒資料。")
        elif not _warn_if_empty(df):
            render_dataset_metrics(df)
            st.caption(f"目前使用：{source_label}")
            if source_path:
                with st.expander("技術資訊", expanded=False):
                    st.caption(f"資料檔：`{_display_path(source_path)}`")

            roles = _infer_column_roles(df)
            st.markdown("### 選對圖種")
            st.caption("每題從下拉選出對應圖種；答對後「用我的資料畫圖」才會啟用。")

            for index, item in enumerate(QUIZ_ITEMS, start=1):
                choice_key = _quiz_choice_key(item.id)
                prev_key = f"chart_quiz_prev_{item.id}"
                if choice_key not in st.session_state:
                    st.session_state[choice_key] = PLEASE_SELECT
                if prev_key not in st.session_state:
                    st.session_state[prev_key] = PLEASE_SELECT

                with st.container(border=True):
                    q_col, a_col, b_col = st.columns([4, 2, 2])
                    q_col.markdown(f"**{index}. {item.prompt}**")
                    choice = a_col.selectbox(
                        "圖種",
                        [PLEASE_SELECT] + CHART_LABELS,
                        key=choice_key,
                        label_visibility="collapsed",
                    )
                    is_correct = _selection_is_correct(item, choice)
                    previous = st.session_state[prev_key]
                    if choice != previous:
                        st.session_state["chart_quiz_focus"] = item.id
                        if not is_correct:
                            st.session_state[_quiz_drawn_key(item.id)] = False
                            if st.session_state.get("chart_quiz_active_draw") == item.id:
                                st.session_state["chart_quiz_active_draw"] = None
                        st.session_state[prev_key] = choice

                    draw_clicked = b_col.button(
                        "用我的資料畫圖",
                        key=f"chart_quiz_draw_{item.id}",
                        disabled=not is_correct,
                        use_container_width=True,
                    )
                    if draw_clicked and is_correct:
                        st.session_state[_quiz_drawn_key(item.id)] = True
                        st.session_state["chart_quiz_focus"] = item.id
                        st.session_state["chart_quiz_active_draw"] = item.id

                    if choice != PLEASE_SELECT and not is_correct:
                        st.caption("目前選擇與這題建議圖種不符，可問右側 Agent 為什麼。")
                    elif choice == PLEASE_SELECT:
                        st.caption("請先選擇圖種；選對後即可畫圖。")
                    else:
                        st.caption("已選對建議圖種，可按「用我的資料畫圖」。")

            active_id = st.session_state.get("chart_quiz_active_draw")
            active_item = next((entry for entry in QUIZ_ITEMS if entry.id == active_id), None)
            if active_item is not None:
                choice = st.session_state.get(_quiz_choice_key(active_item.id), PLEASE_SELECT)
                if _selection_is_correct(active_item, choice) and st.session_state.get(
                    _quiz_drawn_key(active_item.id), False
                ):
                    st.divider()
                    draw_summary_for_agent = _render_quiz_preview(
                        df,
                        active_item,
                        roles,
                        source_label,
                    )
                else:
                    st.session_state["chart_quiz_active_draw"] = None

            with st.expander("自行選圖（進階）", expanded=False):
                st.caption("不走測驗流程時，可在此自由選圖種與欄位。")
                _render_advanced_charts(df, source_label)

    with side:
        focus_id = st.session_state.get("chart_quiz_focus")
        active_draw = st.session_state.get("chart_quiz_active_draw")
        extra_context = _build_agent_context(
            source_label=source_label_for_agent,
            focus_id=focus_id,
            drawn_id=active_draw,
            draw_summary=draw_summary_for_agent,
        )
        st.markdown("##### 建議問 Agent")
        for question in _quiz_agent_hints(focus_id):
            st.markdown(f"- {question}")
        if draw_summary_for_agent:
            st.markdown("- 請依目前預覽圖與數據表，說明圖在比較什麼、最顯著的發現是什麼。")
        render_chat_panel(extra_context=extra_context, page_name="圖表探索")


if os.environ.get("DATASET_CHARTS_SKIP_RUN") != "1":
    run_page()
