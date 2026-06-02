from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.data_ui import (
    CLEANING_LOG_PATH,
    READY_DATASET_PATH,
    SHELL_ROOT,
    WORKING_DATASET_PATH,
    _display_path,
    append_cleaning_log,
    create_ready_dataset,
    load_cleaning_log,
    load_ready_dataset,
    load_working_dataset,
    refresh_working_dataset_cache,
    render_chat_panel,
    render_dataset_metrics,
    reset_working_dataset_from_source,
)
from dataset_streamlit_shell.regression_model import (
    GradientDescentStep,
    LinearModelArtifact,
    apply_standard_scaler,
    create_standard_scaler,
    gradient_descent_steps,
    predict_from_artifact,
    predict_with_parameters,
    save_model_artifact,
)


PromptList = list[str]
_PACKAGE_DIR = Path(__file__).resolve().parent
CORRELATION_FORMULA_IMAGE_PATH = _PACKAGE_DIR / "assets" / "correlation_formula.png"
CATEGORICAL_SELECTION_STATE_KEY = "confirmed_categorical_columns"
CATEGORICAL_SELECTION_WIDGET_KEY = "selected_categorical_columns_widget"
CATEGORICAL_SELECTION_EDIT_WIDGET_KEY = "selected_categorical_columns_edit_widget"
CORRELATION_SELECTION_STATE_KEY = "confirmed_correlation_columns"
CORRELATION_SELECTION_WIDGET_KEY = "selected_correlation_columns_widget"
OUTLIER_COLUMNS_WIDGET_KEY = "outlier_check_columns_widget"
OUTLIER_METHOD_WIDGET_KEY = "outlier_method"
OUTLIER_ZSCORE_THRESHOLD_WIDGET_KEY = "outlier_zscore_threshold"
FEATURE_SCALING_METHOD_WIDGET_KEY = "feature_scaling_method_widget"
FEATURE_SCALING_COLUMNS_WIDGET_KEY = "feature_scaling_columns_widget"

_SCALING_METHOD_LABELS: dict[str, str] = {
    "zscore": "Z 分數正規化（Z-score normalization）",
    "minmax": "最小-最大正規化（Min-Max normalization）",
    "mean_norm": "平均值正規化（Mean normalization）",
    "norm": "正規化（Normalization）",
}
_SCALING_LABEL_TO_METHOD = {label: key for key, label in _SCALING_METHOD_LABELS.items()}
_SCALING_SUFFIXES = {
    "norm": "_norm",
    "minmax": "_minmax",
    "mean_norm": "_mean_norm",
    "zscore": "_z",
}


def _page_shell(
    title: str,
    caption: str,
    render_main: Callable[[pd.DataFrame], None],
    extra_context_builder: Callable[[pd.DataFrame], str] | None = None,
) -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(title)
        st.caption(caption)
        st.info(
            f"目前整理基準：`{_display_path(WORKING_DATASET_PATH)}`。"
            "左側負責診斷與驗證；需要修改資料時，請在右側請 Agent 協作。"
        )
        df = load_working_dataset()
        if df is None:
            st.warning("尚未建立工作資料。請先到「資料上傳與預覽」上傳 CSV。")
            return
        _render_refresh_controls()
        render_main(df)
        _render_recent_log()
    with side:
        extra_context = extra_context_builder(df) if extra_context_builder else ""
        render_chat_panel(extra_context=extra_context, page_name=title)


def _render_refresh_controls() -> None:
    refresh_col, reset_col = st.columns(2)
    if refresh_col.button("重新讀取工作資料", use_container_width=True):
        refresh_working_dataset_cache()
        st.rerun()
    if reset_col.button("回到原始資料", use_container_width=True):
        if reset_working_dataset_from_source():
            st.success("已用原始資料重建工作資料。")
            st.rerun()
        else:
            st.error("找不到原始資料，無法重置。")


def _render_prompts(prompts: PromptList) -> None:
    st.markdown("##### 建議問 Agent")
    st.caption("學生可以自然提問；系統規則會讓 Agent 預設修改工作資料並保護原始資料。")
    for prompt in prompts:
        st.code(prompt, language="text")


def _render_recent_log() -> None:
    with st.expander("最近整理紀錄", expanded=False):
        entries = load_cleaning_log()
        if not entries:
            st.caption(f"尚無紀錄。Agent 修改資料後可寫入 `{_display_path(CLEANING_LOG_PATH)}`。")
            return
        for entry in entries:
            created_at = _format_log_time(str(entry.get("created_at", "")))
            actor = _actor_label(str(entry.get("actor", "")))
            note = str(entry.get("note", ""))
            action = _action_label(str(entry.get("action", "")), note)
            columns = _summarize_columns(entry.get("columns", []))
            rows = entry.get("rows")
            st.markdown(f"**{created_at} · {actor} · {action}**")
            detail = []
            if columns:
                detail.append(f"欄位：{', '.join(str(c) for c in columns)}")
            if rows is not None:
                detail.append(f"筆數：{rows}")
            if note:
                detail.append(note)
            if detail:
                st.caption("；".join(detail))


def _format_log_time(value: str) -> str:
    if not value:
        return "時間不明"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _actor_label(value: str) -> str:
    labels = {"agent": "Agent", "ui": "UI"}
    return labels.get(value.lower(), value or "來源不明")


def _action_label(value: str, note: str = "") -> str:
    labels = {
        "rename_columns_traditional_chinese": "欄位改成繁體中文",
        "rename_columns_to_traditional_chinese": "欄位改成繁體中文",
        "reset_working_dataset": "重置工作資料",
        "create_ready_dataset": "建立分析就緒資料",
        "remove_duplicate_rows": "刪除重複資料列",
        "drop_duplicate_rows": "刪除重複資料列",
        "fill_missing_values": "處理缺失值",
        "fill_missing_age": "補齊年齡空值",
        "handle_outliers": "處理離群值",
        "drop_columns": "刪除欄位",
        "encode_categorical_columns": "類別欄位編碼",
        "feature_scaling": "特徵縮放",
        "add_scaled_columns": "新增縮放欄位",
    }
    normalized = value.strip().lower()
    if normalized in labels:
        return labels[normalized]
    if note:
        return "整理 Working 工作資料"
    return "未命名整理"


def _summarize_columns(value: object, *, max_items: int = 4) -> list[str]:
    if not isinstance(value, list):
        return []
    columns = [str(column) for column in value]
    if len(columns) <= max_items:
        return columns
    remaining = len(columns) - max_items
    return columns[:max_items] + [f"+ {remaining} 個"]


def render_quality_page() -> None:
    def body(df: pd.DataFrame) -> None:
        render_dataset_metrics(df)
        st.markdown("##### 診斷：欄位與資料概覽")
        c1, c2, c3 = st.columns(3)
        c1.metric("重複列", f"{int(df.duplicated().sum()):,}")
        c2.metric("缺失儲存格", f"{int(df.isna().sum().sum()):,}")
        c3.metric("物件/文字欄位", f"{len(df.select_dtypes(include=['object', 'string']).columns):,}")
        overview = pd.DataFrame(
            {
                "資料型態": [str(df[column].dtype) for column in df.columns],
                "非空值筆數": df.notna().sum(),
                "空值筆數": df.isna().sum(),
                "不同值數量": df.nunique(dropna=True),
            },
            index=df.columns,
        )
        st.dataframe(overview, use_container_width=True)
        with st.expander("資料預覽", expanded=False):
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        _render_prompts(
            [
                "請檢查目前工作資料的欄位名稱，並建議哪些欄位需要重新命名。",
                "請檢查目前工作資料的欄位型態是否合理，先不要修改資料。",
                "請把不清楚的欄位名稱改成適合資料分析的名稱，並回報修改前後對照。",
            ]
        )

    _page_shell("欄位與資料概覽", "先看欄位名稱、型態、列數欄數與基本結構。", body)


def render_missing_page() -> None:
    def body(df: pd.DataFrame) -> None:
        missing = df.isna().sum().sort_values(ascending=False)
        missing_frame = pd.DataFrame(
            {
                "空值筆數": missing,
                "空值比例": (missing / max(len(df), 1)).round(4),
                "欄位類型": [_column_kind(df[column]) for column in missing.index],
                "資料型態": [str(df[column].dtype) for column in missing.index],
            }
        )
        st.markdown("##### 診斷：缺失值")
        missing_total = int(df.isna().sum().sum())
        st.metric("缺失儲存格", f"{missing_total:,}")
        if missing_total:
            st.error(
                f"紅燈：目前還有 {missing_total:,} 個缺失儲存格，"
                "建議先請 Agent 處理後再建立 Ready 分析就緒資料。"
            )
        else:
            st.success("綠燈：目前沒有缺失儲存格，可以進入下一個整理步驟。")
        st.dataframe(missing_frame, use_container_width=True)
        _render_prompts(
            [
                "請依缺失比例整理目前工作資料的缺失值問題，先不要修改資料。",
                "請建議各欄位缺失值適合刪除、補平均數、中位數、眾數，或另外建立 Unknown 類別。",
                "請依你的建議處理目前工作資料的缺失值，並回報每個欄位修改了幾筆。",
            ]
        )

    _page_shell("缺失值處理", "專心判斷缺失值，不混入離群值或分布探索。", body)


def _column_kind(series: pd.Series) -> str:
    column_name = str(series.name or "")
    normalized_name = column_name.lower()
    non_null = series.dropna()

    if _looks_like_identifier(normalized_name, non_null):
        return "識別欄位"

    if not pd.api.types.is_numeric_dtype(series):
        return "類別"

    unique_count = int(non_null.nunique())
    if unique_count <= 2:
        return "類別"

    if _looks_like_coded_category(normalized_name, len(series), unique_count):
        return "疑似類別（數字代碼）"

    return "數值"


def _looks_like_identifier(normalized_name: str, non_null: pd.Series) -> bool:
    id_keywords = ["id", "編號", "序號", "流水號", "識別"]
    if not any(keyword in normalized_name for keyword in id_keywords):
        return False
    return int(non_null.nunique()) >= max(int(len(non_null) * 0.8), 1)


def _looks_like_coded_category(
    normalized_name: str,
    row_count: int,
    unique_count: int,
) -> bool:
    category_keywords = [
        "class",
        "pclass",
        "level",
        "grade",
        "rank",
        "type",
        "category",
        "艙等",
        "等級",
        "類別",
        "分類",
    ]
    if any(keyword in normalized_name for keyword in category_keywords):
        return True
    unique_ratio = unique_count / max(row_count, 1)
    return unique_count <= 10 and unique_ratio <= 0.05


def _teaching_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if _column_kind(df[column]) in {"類別", "疑似類別（數字代碼）"}
    ]


def _selected_categorical_columns(df: pd.DataFrame) -> list[str]:
    selected = st.session_state.get(CATEGORICAL_SELECTION_STATE_KEY, [])
    if not isinstance(selected, list):
        return []
    all_columns = {str(column) for column in df.columns}
    return [str(column) for column in selected if str(column) in all_columns]


def _set_selected_categorical_columns(df: pd.DataFrame, selected: list[str]) -> list[str]:
    all_columns = {str(column) for column in df.columns}
    cleaned = [str(column) for column in selected if str(column) in all_columns]
    st.session_state[CATEGORICAL_SELECTION_STATE_KEY] = cleaned
    return cleaned


def _categorical_extra_context(df: pd.DataFrame) -> str:
    selected = _selected_categorical_columns(df)
    if not selected:
        return "目前學生尚未在 UI 中確認類別欄位。"
    return "目前學生在 UI 中確認的類別欄位：" + "、".join(selected) + "。"


def _selected_correlation_columns(df: pd.DataFrame) -> list[str]:
    selected = st.session_state.get(CORRELATION_SELECTION_STATE_KEY, [])
    if not isinstance(selected, list):
        return []
    all_columns = {str(column) for column in df.columns}
    return [str(column) for column in selected if str(column) in all_columns]


def _set_selected_correlation_columns(df: pd.DataFrame, selected: list[str]) -> list[str]:
    all_columns = {str(column) for column in df.columns}
    cleaned = [str(column) for column in selected if str(column) in all_columns]
    st.session_state[CORRELATION_SELECTION_STATE_KEY] = cleaned
    return cleaned


def _correlation_extra_context(df: pd.DataFrame) -> str:
    selected = _selected_correlation_columns(df)
    if not selected:
        return "目前學生尚未在 UI 中確認要做數值相關性的欄位。"
    return "目前學生在 UI 中確認要做數值相關性的欄位：" + "、".join(selected) + "。"


def _duplicate_rule_columns(df: pd.DataFrame) -> list[str]:
    selected = st.session_state.get("duplicate_rule_columns", [])
    if not isinstance(selected, list):
        return []
    all_columns = {str(column) for column in df.columns}
    return [str(column) for column in selected if str(column) in all_columns]


def _duplicates_extra_context(df: pd.DataFrame) -> str:
    selected = _duplicate_rule_columns(df)
    if not selected:
        return "目前學生以「整列完全相同」作為重複定義。"
    return "目前學生選擇以這些欄位判斷重複：" + "、".join(selected) + "。"


def _selected_outlier_columns(df: pd.DataFrame) -> list[str]:
    selected = st.session_state.get(OUTLIER_COLUMNS_WIDGET_KEY, [])
    if not isinstance(selected, list):
        return []
    allowed = set(_numeric_dtype_columns(df))
    return [str(column) for column in selected if str(column) in allowed]


def _outliers_extra_context(df: pd.DataFrame) -> str:
    selected = _selected_outlier_columns(df)
    method = st.session_state.get(OUTLIER_METHOD_WIDGET_KEY, "請選擇方法")
    parts = ["目前頁面：離群值檢查。"]
    if not selected:
        parts.append("學生尚未在 UI 中選擇要檢查的數值欄位。")
    else:
        parts.append("學生在 UI 中選擇要檢查的數值欄位：" + "、".join(selected) + "。")
    if not isinstance(method, str) or method == "請選擇方法":
        parts.append("尚未選擇離群值判斷方法。")
    elif method == "IQR 法":
        parts.append("學生選擇的離群值判斷方法：IQR 法（1.5 × IQR 規則）。")
    elif method == "Z-score 法":
        threshold = st.session_state.get(OUTLIER_ZSCORE_THRESHOLD_WIDGET_KEY, 3.0)
        parts.append(f"學生選擇的離群值判斷方法：Z-score 法，閾值 {threshold}。")
    parts.append("請依學生選定的欄位與方法討論離群值；若要修改資料，請寫入 cleaning_log.jsonl。")
    return "".join(parts)


def _column_overview(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "資料型態": [str(df[column].dtype) for column in columns],
            "空值筆數": df[columns].isna().sum(),
            "不同值數量": df[columns].nunique(dropna=True),
            "常見值": [_safe_top_value(df[column]) for column in columns],
        }
    )


def _encoding_preview_columns(df: pd.DataFrame, categorical: list[str]) -> list[str]:
    all_columns = [str(column) for column in df.columns]
    preview_columns: list[str] = []
    for column in categorical:
        related = [
            candidate
            for candidate in all_columns
            if candidate == column or candidate.startswith(f"{column}_")
        ]
        for candidate in related:
            if candidate not in preview_columns:
                preview_columns.append(candidate)
    return preview_columns


def _display_encoding_preview(df: pd.DataFrame) -> pd.DataFrame:
    preview = df.copy()
    bool_columns = preview.select_dtypes(include="bool").columns
    for column in bool_columns:
        preview[column] = preview[column].astype(int)
    return preview


def _safe_top_value(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def _category_kind_hint(kind: str) -> str:
    if kind == "疑似類別（數字代碼）":
        return "這個欄位雖然是數字，但不同值很少或名稱像等級/類別，建議先當類別理解。"
    return "文字、布林或狀態欄位，適合先當類別理解。"


def render_duplicates_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：刪除重複資料列")
        st.caption(
            "先定義什麼算重複。未選欄位時，預設每個欄位都相同才算同一筆；"
            "選多個欄位時，必須所有選取欄位都相同，才會歸在同一個重複組別。"
        )
        all_columns = [str(column) for column in df.columns]
        selected_columns = st.multiselect(
            "用哪些欄位判斷重複",
            all_columns,
            default=[],
            placeholder="不選欄位時，使用整列完全相同判斷",
            key="duplicate_rule_columns",
        )
        rule_columns = selected_columns or all_columns
        rule_label = "整列完全相同" if not selected_columns else " + ".join(selected_columns)

        duplicated_mask = df.duplicated(subset=rule_columns, keep=False)
        duplicate_candidates = df.loc[duplicated_mask].copy()
        duplicate_groups = 0
        estimated_delete = 0

        if not duplicate_candidates.empty:
            grouped = duplicate_candidates.groupby(rule_columns, dropna=False, sort=False)
            duplicate_candidates.insert(0, "重複組別", grouped.ngroup() + 1)
            duplicate_candidates.insert(1, "組內序號", grouped.cumcount() + 1)
            duplicate_candidates.insert(
                2,
                "建議處理",
                duplicate_candidates["組內序號"].map(
                    lambda index: "保留第一筆" if index == 1 else "可請 Agent 判斷刪除"
                ),
            )
            duplicate_groups = int(duplicate_candidates["重複組別"].nunique())
            estimated_delete = len(duplicate_candidates) - duplicate_groups

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("總資料列數", f"{len(df):,}")
        c2.metric("重複候選列數", f"{len(duplicate_candidates):,}")
        c3.metric("重複組別數", f"{duplicate_groups:,}")
        c4.metric("預估可刪除筆數", f"{estimated_delete:,}")

        st.markdown("##### 目前重複定義")
        st.write(rule_label)

        if duplicate_candidates.empty:
            st.success("依照目前定義，沒有找到重複資料列。")
        else:
            st.markdown("##### 重複候選預覽")
            st.caption(
                "`重複組別` 相同，代表這些資料依照目前規則被判定為同一組疑似重複資料。"
                "這裡只預覽前 50 筆，真正刪除仍請右側 Agent 執行。"
            )
            st.dataframe(
                duplicate_candidates.head(50),
                use_container_width=True,
                hide_index=True,
            )

        rule_for_prompt = (
            "整列完全相同"
            if not selected_columns
            else "、".join(f"`{column}`" for column in selected_columns)
        )
        _render_prompts(
            [
                f"請依「{rule_for_prompt}」檢查目前工作資料的重複資料列，先不要修改資料，請說明重複候選列數與可能影響。",
                f"請依「{rule_for_prompt}」刪除目前工作資料中的重複資料列，每個重複組別保留第一筆，並回報刪除了幾筆。",
                "請刪除重複資料後，在 cleaning_log.jsonl 追加一筆紀錄，actor 是 agent，action 是 remove_duplicate_rows。",
            ]
        )

    _page_shell(
        "刪除重複資料列",
        "讓學生先定義重複規則，再請 Agent 刪除重複列。",
        body,
        extra_context_builder=_duplicates_extra_context,
    )


def _numeric_dtype_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _outlier_column_overview(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "資料型態": [str(df[column].dtype) for column in columns],
            "欄位類型（輔助）": [_column_kind(df[column]) for column in columns],
            "空值筆數": df[columns].isna().sum(),
            "不同值數量": df[columns].nunique(dropna=True),
        },
        index=columns,
    )


def render_outliers_page() -> None:
    def body(df: pd.DataFrame) -> None:
        numeric_columns = _numeric_dtype_columns(df)
        st.markdown("##### 診斷：離群值")
        if not numeric_columns:
            st.warning("目前沒有數值欄位。請先完成欄位整理或編碼。")
            return

        st.info(
            "請先勾選要檢查的數值欄位，再選擇離群值判斷方法。"
            "欄位類型僅供參考，不會自動替你排除；SibSp、Parch 這類計數欄位也可自行選取。"
        )
        st.markdown("###### 數值欄位輔助資訊")
        st.dataframe(
            _outlier_column_overview(df, numeric_columns),
            use_container_width=True,
        )

        selected_columns = st.multiselect(
            "選擇要檢查離群值的數值欄位",
            numeric_columns,
            key=OUTLIER_COLUMNS_WIDGET_KEY,
        )
        if not selected_columns:
            st.warning("請至少選擇一個數值欄位。")
            return

        numeric = df[selected_columns].apply(pd.to_numeric, errors="coerce")

        method = st.selectbox(
            "離群值判斷方法",
            ["請選擇方法", "IQR 法", "Z-score 法"],
            key=OUTLIER_METHOD_WIDGET_KEY,
        )
        if method == "請選擇方法":
            st.info("請先選擇離群值判斷方法。")
            return

        if method == "IQR 法":
            outlier_frame = _iqr_outlier_summary(df, numeric)
        else:
            threshold = st.slider(
                "Z-score 閾值",
                min_value=2.0,
                max_value=4.0,
                value=3.0,
                step=0.1,
                key=OUTLIER_ZSCORE_THRESHOLD_WIDGET_KEY,
            )
            outlier_frame = _zscore_outlier_summary(df, numeric, threshold)

        if outlier_frame.empty:
            st.warning("目前選取的欄位沒有足夠資料可檢查離群值。")
            return

        st.markdown("##### 離群值檢查結果")
        st.dataframe(outlier_frame, use_container_width=True, hide_index=True)
        outlier_columns = outlier_frame[outlier_frame["離群值筆數"] > 0]["欄位名稱"].tolist()
        if not outlier_columns:
            st.success("依照目前方法，沒有偵測到離群值欄位。")
            return

        selected = st.selectbox(
            "選擇欄位查看細節",
            ["請選擇欄位"] + outlier_columns,
            key="outlier_detail_column",
        )
        if selected == "請選擇欄位":
            st.info("請選擇一個欄位，查看圖形與離群值資料列。")
            return

        selected_rule = outlier_frame[outlier_frame["欄位名稱"] == selected].iloc[0]
        selected_values = pd.to_numeric(df[selected], errors="coerce")
        if method == "IQR 法":
            outlier_mask = (selected_values < selected_rule["離群值下界"]) | (
                selected_values > selected_rule["離群值上界"]
            )
            _render_iqr_outlier_chart(selected_values, selected)
        else:
            mean = float(selected_rule["平均數"])
            std = float(selected_rule["標準差"])
            threshold_value = float(selected_rule["Z-score 閾值"])
            zscores = (selected_values - mean) / std if std else selected_values * 0
            outlier_mask = zscores.abs() > threshold_value
            _render_zscore_outlier_chart(selected_values, selected, mean, std, threshold_value)

        outlier_rows = df[outlier_mask.fillna(False)]
        st.markdown("##### 離群值資料列預覽")
        st.dataframe(outlier_rows.head(30), use_container_width=True, hide_index=True)
        _render_prompts(
            [
                f"請使用 {method} 檢查目前工作資料的 `{selected}` 欄位離群值，先不要修改資料。",
                f"請針對 `{selected}` 欄位說明這些離群值可能是錯誤資料，還是真實但極端的觀察。",
                f"請依你的判斷處理 `{selected}` 欄位的離群值，並回報修改前後摘要，最後寫入 cleaning_log.jsonl。",
            ]
        )

    _page_shell(
        "離群值檢查",
        "專心檢查極端數值，避免和缺失值、分布探索混在一起。",
        body,
        extra_context_builder=_outliers_extra_context,
    )


def _iqr_outlier_summary(df: pd.DataFrame, numeric: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "欄位名稱": str(column),
                "離群值筆數": outlier_count,
                "離群值比例": round(outlier_count / max(len(df), 1), 4),
                "第一四分位數": q1,
                "第三四分位數": q3,
                "四分位距": iqr,
                "離群值下界": lower,
                "離群值上界": upper,
            }
        )
    return pd.DataFrame(rows).sort_values("離群值筆數", ascending=False)


def _zscore_outlier_summary(
    df: pd.DataFrame,
    numeric: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) < 2:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0:
            outlier_count = 0
        else:
            zscores = (values - mean) / std
            outlier_count = int((zscores.abs() > threshold).sum())
        rows.append(
            {
                "欄位名稱": str(column),
                "離群值筆數": outlier_count,
                "離群值比例": round(outlier_count / max(len(df), 1), 4),
                "平均數": mean,
                "標準差": std,
                "Z-score 閾值": threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("離群值筆數", ascending=False)


def _render_iqr_outlier_chart(values: pd.Series, column: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
    ax.boxplot(values.dropna(), vert=False, patch_artist=True)
    ax.set_title(f"{column} 的箱形圖")
    ax.set_xlabel(column)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_zscore_outlier_chart(
    values: pd.Series,
    column: str,
    mean: float,
    std: float,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    clean_values = values.dropna()
    ax.hist(clean_values, bins=24, alpha=0.75)
    lower = mean - threshold * std
    upper = mean + threshold * std
    ax.axvline(lower, color="red", linestyle="--", label="離群值下界")
    ax.axvline(upper, color="red", linestyle="--", label="離群值上界")
    ax.set_title(f"{column} 的 Z-score 分布")
    ax.set_xlabel(column)
    ax.set_ylabel("筆數")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_numeric_page() -> None:
    render_outliers_page()


def render_categorical_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：類別欄位")
        st.info(
            "類別欄位需要根據資料意義判斷，不只看資料型態。"
            "請先和 Agent 討論哪些欄位應視為類別欄位，再在下方選取欄位查看分布。"
        )
        all_columns = [str(column) for column in df.columns]
        st.markdown("###### 欄位輔助資訊")
        st.dataframe(_column_overview(df, all_columns), use_container_width=True)

        selected_columns = st.multiselect(
            "請選擇你和 Agent 確認的類別欄位",
            all_columns,
            default=_selected_categorical_columns(df),
            key=CATEGORICAL_SELECTION_WIDGET_KEY,
        )
        selected_columns = _set_selected_categorical_columns(df, selected_columns)
        if not selected_columns:
            st.warning("尚未選擇類別欄位。請先和 Agent 討論，再勾選你確認的欄位。")
        else:
            st.markdown("###### 已確認的類別欄位")
            st.dataframe(_column_overview(df, selected_columns), use_container_width=True)
            selected = st.selectbox(
                "查看類別分布",
                ["請選擇欄位"] + selected_columns,
                key="categorical_distribution_column",
            )
            if selected != "請選擇欄位":
                counts = df[selected].fillna("Missing").astype(str).value_counts().head(30)
                st.bar_chart(counts)
        _render_prompts(
            [
                "請根據目前工作資料，判斷哪些欄位適合視為類別欄位，並逐一說明理由。",
                "請指出哪些數字欄位其實可能是類別代碼，哪些數字欄位比較像真正的數值或計數。",
                "請根據我選出的類別欄位，檢查是否有需要合併稀有類別或補上 Unknown 的欄位。",
            ]
        )

    _page_shell(
        "類別欄位診斷",
        "由學生和 Agent 協作確認哪些欄位應視為類別。",
        body,
        extra_context_builder=_categorical_extra_context,
    )


def render_encoding_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：類別欄位編碼")
        all_columns = [str(column) for column in df.columns]
        categorical = _selected_categorical_columns(df)
        if categorical:
            st.success("已套用「類別欄位整理」頁確認的欄位。")
            st.write("目前類別欄位：" + "、".join(categorical))
        else:
            st.warning("尚未選擇類別欄位。請先到「類別欄位整理」頁和 Agent 協作確認。")

        with st.expander("需要修正選取欄位？", expanded=False):
            edited_columns = st.multiselect(
                "修正類別欄位",
                all_columns,
                default=categorical,
                key=CATEGORICAL_SELECTION_EDIT_WIDGET_KEY,
            )
            categorical = _set_selected_categorical_columns(df, edited_columns)

        if categorical:
            st.markdown("###### 準備編碼的類別欄位")
            overview = _column_overview(df, categorical)
            overview["編碼提醒"] = [
                "先和 Agent 確認要使用 One-Hot、Label Encoding，或保留原欄位。"
                for _ in categorical
            ]
            st.dataframe(overview, use_container_width=True)

            preview_columns = _encoding_preview_columns(df, categorical)
            st.markdown("###### 目前工作資料預覽：類別欄位與編碼結果")
            st.caption(
                "請 Agent 完成編碼後，按「重新讀取工作資料」。"
                "這裡會顯示原類別欄位與可能新增的編碼欄位，方便檢查 0/1 或數字編碼結果。"
            )
            if preview_columns:
                st.dataframe(
                    _display_encoding_preview(df[preview_columns].head(20)),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("目前找不到可預覽的類別欄位或編碼結果欄位。")
        _render_prompts(
            [
                "請根據目前已選取的類別欄位，建議哪些欄位適合 One-Hot Encoding，哪些適合 Label Encoding。",
                "請先不要修改資料，說明哪些欄位適合做 One-Hot Encoding，哪些不適合。",
                "請針對適合的類別欄位新增 One-Hot Encoding 欄位，保留原欄位，並回報新增了哪些欄位。",
            ]
        )

    _page_shell(
        "類別欄位編碼",
        "把類別欄位轉成後續分析可用的數值表示。",
        body,
        extra_context_builder=_categorical_extra_context,
    )


def _render_correlation_formula_reference() -> None:
    with st.expander("共變異數與相關係數公式", expanded=False):
        if CORRELATION_FORMULA_IMAGE_PATH.is_file():
            st.image(str(CORRELATION_FORMULA_IMAGE_PATH), use_container_width=True)
        else:
            st.caption("公式說明圖尚未就緒。")
        st.caption("本頁相關性矩陣每格為 Pearson 相關係數 r，範圍 [-1, 1]。")


def render_correlation_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：數值相關性")
        st.info(
            "請先和 Agent 討論哪些欄位適合做數值相關性分析，再在下方選取欄位。"
            "選取至少兩個欄位後，系統只顯示這些欄位之間的完整相關矩陣。"
        )
        _render_correlation_formula_reference()
        all_columns = [str(column) for column in df.columns]
        st.markdown("###### 欄位輔助資訊")
        st.dataframe(_column_overview(df, all_columns), use_container_width=True)

        selected_columns = st.multiselect(
            "請選擇你和 Agent 確認要做數值相關性的欄位",
            all_columns,
            default=_selected_correlation_columns(df),
            key=CORRELATION_SELECTION_WIDGET_KEY,
        )
        selected_columns = _set_selected_correlation_columns(df, selected_columns)
        if len(selected_columns) < 2:
            st.warning("請至少選擇兩個欄位，才能計算相關性矩陣。")
            return

        numeric_frame = df[selected_columns].apply(pd.to_numeric, errors="coerce")
        usable_columns = [
            column for column in selected_columns if numeric_frame[column].notna().sum() >= 2
        ]
        skipped_columns = [column for column in selected_columns if column not in usable_columns]
        if skipped_columns:
            st.warning(
                "以下欄位無法轉成足夠的數值資料，暫不納入相關矩陣："
                + "、".join(skipped_columns)
            )
        if len(usable_columns) < 2:
            st.warning("目前可計算相關性的欄位少於兩個，請重新選擇欄位。")
            return

        corr = numeric_frame[usable_columns].corr()
        st.markdown("###### 相關性矩陣")
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
        _render_prompts(
            [
                "請解讀我目前選取欄位之間的相關性矩陣，指出值得注意的關係。",
                "請根據這些欄位的相關矩陣，判斷是否有欄位可能帶有重複資訊，先不要修改資料。",
                "請說明這些欄位的相關性對後續學習或分析可能有什麼影響。",
            ]
        )

    _page_shell(
        "數值相關性",
        "在建立 Ready 分析就緒資料之前，檢查學生選取欄位之間的數值關係。",
        body,
        extra_context_builder=_correlation_extra_context,
    )


def _scaling_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _is_binary_like_column(series: pd.Series) -> bool:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty or int(non_null.nunique()) > 2:
        return False
    unique_values = set(non_null.unique().tolist())
    return unique_values.issubset({0, 1, 0.0, 1.0})


def _scaled_output_column(column: str, method: str) -> str:
    return f"{column}{_SCALING_SUFFIXES[method]}"


def _scaling_method_detail(method: str) -> tuple[str, str]:
    details = {
        "norm": (
            "$x' = x / x_{max}$",
            "結果約在 $[0, 1]$；**$x$ 必須 $\\ge 0$**；$x_{max} \\ne 0$。",
        ),
        "minmax": (
            "$x' = \\dfrac{x - x_{min}}{x_{max} - x_{min}}$",
            "結果約在 $[0, 1]$；$x_{max} \\ne x_{min}$。",
        ),
        "mean_norm": (
            "$x' = \\dfrac{x - \\mu}{x_{max} - x_{min}}$",
            "結果約在 $[-1, 1]$；$x_{max} \\ne x_{min}$。",
        ),
        "zscore": (
            "$x' = \\dfrac{x - \\mu}{\\sigma}$",
            "實務上約在 $[-3, 3]$；$\\sigma \\ne 0$；適合 PCA、分群、KNN、SVM。",
        ),
    }
    return details[method]


def _render_scaling_formula_reference() -> None:
    with st.expander("四種方法公式對照", expanded=False):
        st.markdown(
            """
| 方法 | 公式 | 前提／範圍 |
|---|---|---|
| 正規化 | $x' = x / x_{max}$ | $x \\ge 0$；$0 \\le x' \\le 1$ |
| 最小-最大正規化 | $x' = \\dfrac{x - x_{min}}{x_{max} - x_{min}}$ | $0 \\le x' \\le 1$ |
| 平均值正規化 | $x' = \\dfrac{x - \\mu}{x_{max} - x_{min}}$ | 約 $[-1, 1]$ |
| Z 分數正規化 | $x' = \\dfrac{x - \\mu}{\\sigma}$ | 實務約 $[-3, 3]$ |
"""
        )


def _feature_scaling_overview(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            rows.append(
                {
                    "欄位名稱": column,
                    "最小值": None,
                    "最大值": None,
                    "平均 μ": None,
                    "標準差 σ": None,
                    "空值筆數": int(series.isna().sum()),
                    "備註": "無可用數值",
                }
            )
            continue
        min_value = float(non_null.min())
        max_value = float(non_null.max())
        mean_value = float(non_null.mean())
        std_value = float(non_null.std(ddof=0))
        notes: list[str] = []
        if _is_binary_like_column(series):
            notes.append("疑似 0/1 欄位，通常不必縮放")
        if min_value < 0:
            notes.append("含負值，不適用正規化（÷ max）")
        if max_value == min_value:
            notes.append("常數欄，無法 Min-Max／平均值正規化／Z-score")
        elif max_value == 0:
            notes.append("最大值為 0，無法正規化（÷ max）")
        if std_value == 0:
            notes.append("標準差為 0，無法 Z-score")
        rows.append(
            {
                "欄位名稱": column,
                "最小值": round(min_value, 4),
                "最大值": round(max_value, 4),
                "平均 μ": round(mean_value, 4),
                "標準差 σ": round(std_value, 4),
                "空值筆數": int(series.isna().sum()),
                "備註": "；".join(notes) if notes else "可檢查",
            }
        )
    return pd.DataFrame(rows)


def _validate_scaling_column(series: pd.Series, method: str) -> tuple[bool, str]:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return False, "沒有可用數值"
    min_value = float(non_null.min())
    max_value = float(non_null.max())
    std_value = float(non_null.std(ddof=0))
    if method == "norm":
        if min_value < 0:
            return False, "含負值，不適用正規化（÷ max）"
        if max_value == 0:
            return False, "最大值為 0"
        return True, ""
    if max_value == min_value:
        return False, "常數欄，分母為 0"
    if method == "zscore" and std_value == 0:
        return False, "標準差為 0"
    return True, ""


def _apply_scaling(series: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    min_value = float(non_null.min())
    max_value = float(non_null.max())
    mean_value = float(non_null.mean())
    std_value = float(non_null.std(ddof=0))
    if method == "norm":
        return numeric / max_value
    if method == "minmax":
        return (numeric - min_value) / (max_value - min_value)
    if method == "mean_norm":
        return (numeric - mean_value) / (max_value - min_value)
    return (numeric - mean_value) / std_value


def _selected_feature_scaling_method() -> str | None:
    label = st.session_state.get(FEATURE_SCALING_METHOD_WIDGET_KEY)
    if not isinstance(label, str):
        return None
    return _SCALING_LABEL_TO_METHOD.get(label)


def _selected_feature_scaling_columns(df: pd.DataFrame) -> list[str]:
    selected = st.session_state.get(FEATURE_SCALING_COLUMNS_WIDGET_KEY, [])
    if not isinstance(selected, list):
        return []
    allowed = set(_scaling_numeric_columns(df))
    return [str(column) for column in selected if str(column) in allowed]


def _feature_scaling_extra_context(df: pd.DataFrame) -> str:
    method = _selected_feature_scaling_method()
    selected = _selected_feature_scaling_columns(df)
    if method is None:
        return "目前學生尚未在 UI 中選擇特徵縮放方法。"
    method_label = _SCALING_METHOD_LABELS[method]
    if not selected:
        return f"目前學生選擇的縮放方法：{method_label}；尚未選擇欄位。"
    valid_columns: list[str] = []
    invalid_notes: list[str] = []
    for column in selected:
        ok, reason = _validate_scaling_column(df[column], method)
        if ok:
            valid_columns.append(column)
        else:
            invalid_notes.append(f"{column}（{reason}）")
    parts = [
        f"目前頁面：特徵縮放（Feature Scaling）。",
        f"學生選擇的縮放方法：{method_label}。",
    ]
    if valid_columns:
        new_names = "、".join(_scaled_output_column(column, method) for column in valid_columns)
        parts.append("學生選擇且符合前提的欄位：" + "、".join(valid_columns) + "。")
        parts.append(f"建議新增欄位名稱：{new_names}。")
    else:
        parts.append("目前選取的欄位都不符合此方法的前提。")
    if invalid_notes:
        parts.append("不適用欄位：" + "；".join(invalid_notes) + "。")
    parts.append("請在 working.csv 新增縮放欄位並保留原欄位，不要修改 ready.csv。")
    return "".join(parts)


def render_feature_scaling_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：特徵縮放（Feature Scaling）")
        st.info(
            "不同數值欄位的量綱可能差很多。請選擇縮放方式與欄位，"
            "由 Agent 在 working 新增欄位；**保留原欄位**，不要覆蓋。"
            "缺失值會略過不參與 μ、σ、min、max 計算；建議先完成缺失值處理。"
        )
        numeric_columns = _scaling_numeric_columns(df)
        if len(numeric_columns) < 1:
            st.warning("目前沒有數值欄位。請先完成欄位整理或編碼。")
            return

        st.markdown("###### 數值欄位概覽")
        st.dataframe(
            _feature_scaling_overview(df, numeric_columns),
            use_container_width=True,
            hide_index=True,
        )

        with st.container(border=True):
            st.markdown("###### 縮放設定")
            method_labels = list(_SCALING_METHOD_LABELS.values())
            default_index = method_labels.index(_SCALING_METHOD_LABELS["zscore"])
            method_label = st.radio(
                "縮放方法",
                method_labels,
                index=default_index,
                key=FEATURE_SCALING_METHOD_WIDGET_KEY,
            )
            method = _SCALING_LABEL_TO_METHOD[method_label]
            formula, note = _scaling_method_detail(method)
            st.markdown(f"**{method_label}**")
            st.markdown(formula)
            st.caption(note)
            _render_scaling_formula_reference()

            selected_columns = st.multiselect(
                "選擇要縮放的數值欄位",
                numeric_columns,
                key=FEATURE_SCALING_COLUMNS_WIDGET_KEY,
            )
            if not selected_columns:
                st.warning("請至少選擇一個數值欄位。")
                return

            valid_columns: list[str] = []
            for column in selected_columns:
                ok, reason = _validate_scaling_column(df[column], method)
                if ok:
                    valid_columns.append(column)
                else:
                    st.warning(f"`{column}`：{reason}")

            suggested = [
                _scaled_output_column(column, method) for column in valid_columns
            ]
            if suggested:
                st.markdown("###### 預計新增欄位")
                st.write("、".join(f"`{name}`" for name in suggested))
                existing = [name for name in suggested if name in df.columns]
                if existing:
                    st.warning(
                        "以下欄位已存在，請 Agent 改用不同後綴或先確認是否覆寫："
                        + "、".join(f"`{name}`" for name in existing)
                    )
            else:
                st.warning("目前選取的欄位都不符合此方法的前提，請調整方法或欄位。")
                return

            preview = df[valid_columns].head(10).copy()
            for column in valid_columns:
                output_column = _scaled_output_column(column, method)
                preview[output_column] = _apply_scaling(df[column], method).head(10)
            st.markdown("###### 預覽（前 10 筆，尚未寫入 working）")
            st.dataframe(preview, use_container_width=True, hide_index=True)

        method_name = _SCALING_METHOD_LABELS[method]
        column_list = "、".join(f"`{column}`" for column in valid_columns)
        new_columns = "、".join(
            f"`{_scaled_output_column(column, method)}`" for column in valid_columns
        )
        _render_prompts(
            [
                f"請依 **{method_name}** 為 {column_list} 在 working 新增縮放欄位（例如 {new_columns}），保留原欄位，並回報新欄名稱。",
                f"請說明 **{method_name}** 為什麼適合或不適合我目前選的欄位。",
                "請將此次修改寫入 cleaning_log，action 使用 feature_scaling。",
            ]
        )

    _page_shell(
        "特徵縮放（Feature Scaling）",
        "選擇縮放方式與欄位，由 Agent 新增縮放欄位；建立 Ready 前完成。",
        body,
        extra_context_builder=_feature_scaling_extra_context,
    )


def text_or_category_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.select_dtypes(include=["object", "string", "category"]).columns
    ]


def render_encoding_correlation_page() -> None:
    render_encoding_page()


def render_ready_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 建立 Ready 分析就緒資料")
        st.caption("將目前 Working 工作資料凍結為穩定的 `ready.csv`，供後續學習、分析與訓練使用。")
        missing_total = int(df.isna().sum().sum())
        text_columns = text_or_category_columns(df)
        duplicate_rows = int(df.duplicated().sum())
        numeric_cols = len(df.select_dtypes(include="number").columns)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("列數", f"{len(df):,}")
        c2.metric("數值欄位", f"{numeric_cols:,}")
        c3.metric("缺失儲存格", f"{missing_total:,}")
        c4.metric("重複列", f"{duplicate_rows:,}")
        if text_columns:
            st.warning(
                f"仍有 {len(text_columns)} 個文字/類別欄位："
                + "、".join(f"`{column}`" for column in text_columns)
                + "。後續分析或訓練可能需要先做編碼，或在分析頁只選數值欄位。"
            )
        if missing_total:
            st.warning("仍有缺失值。後續分析或訓練前建議先完成缺失值處理。")
        if st.button("建立 ready.csv", type="primary", use_container_width=True):
            create_ready_dataset(df)
            append_cleaning_log(
                action="create_ready_dataset",
                columns=df.columns,
                rows=len(df),
                note="由 working.csv 凍結為 ready.csv。",
                actor="ui",
            )
            st.success(f"已建立 `{_display_path(READY_DATASET_PATH)}`。")
        ready = load_ready_dataset()
        if ready is not None:
            st.markdown("###### 目前 Ready 分析就緒資料")
            render_dataset_metrics(ready)
            st.download_button(
                "下載 ready.csv",
                data=ready.to_csv(index=False).encode("utf-8-sig"),
                file_name="ready.csv",
                mime="text/csv",
                use_container_width=True,
            )
        _render_prompts(
            [
                "請檢查目前 Working 工作資料是否適合建立 Ready 分析就緒資料，列出還需要整理的問題。",
                "請確認目前工作資料是否還有缺失值、重複列或未編碼欄位，先不要修改資料。",
                "請建議建立 Ready 分析就緒資料前還需要完成哪些整理步驟。",
            ]
        )

    _page_shell("建立 Ready 分析就緒資料", "把 Working 工作資料凍結成後續分析使用的穩定資料表。", body)


def pca_status(df: pd.DataFrame) -> dict[str, object]:
    numeric = df.select_dtypes(include="number")
    return {
        "rows": len(df),
        "numeric_columns": len(numeric.columns),
        "missing_cells": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        "enough_columns": len(numeric.columns) >= 2,
        "enough_rows": len(df) >= 3,
    }


def render_analysis_shell(title: str, caption: str, render_main: Callable[[pd.DataFrame], None]) -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(title)
        st.caption(caption)
        st.info(f"目前分析基準：Ready 分析就緒資料 `{_display_path(READY_DATASET_PATH)}`。")
        df = load_ready_dataset()
        if df is None:
            st.warning("尚未建立 Ready 分析就緒資料。請先到「建立 Ready 分析就緒資料」頁完成匯出。")
            return
        render_main(df)
    with side:
        render_chat_panel(page_name=title)


REGRESSION_DEMO_DIR = SHELL_ROOT / "built-in-data" / "regression"
REGRESSION_MODEL_DIR = SHELL_ROOT / "workspace" / "models" / "regression"
RESTAURANT_PROFIT_PATH = REGRESSION_DEMO_DIR / "restaurant_profit.csv"
HOUSE_PRICES_PATH = REGRESSION_DEMO_DIR / "house_prices.csv"


def render_simple_linear_regression_page() -> None:
    def body(df: pd.DataFrame, source_label: str) -> None:
        st.markdown("##### 單變量線性回歸")
        st.info(
            "本頁專注在教材的第一個回歸模型：用一個 feature 預測一個連續 target，"
            "觀察回歸線、預測誤差與成本函數 J(w,b)。"
        )
        numeric_columns = _numeric_regression_columns(df)
        if len(numeric_columns) < 2:
            st.warning("單變量線性回歸至少需要 2 個數值欄位：1 個 feature 與 1 個 target。")
            return

        default_feature = _default_column(numeric_columns, "城市人口_萬人")
        default_target = _default_column(numeric_columns, "餐廳獲利_萬美元", exclude={default_feature})
        c1, c2 = st.columns(2)
        feature = c1.selectbox(
            "選擇 feature（x）",
            numeric_columns,
            index=numeric_columns.index(default_feature),
            key="simple_regression_feature",
        )
        target_options = [column for column in numeric_columns if column != feature]
        target = c2.selectbox(
            "選擇 target（y）",
            target_options,
            index=target_options.index(default_target) if default_target in target_options else 0,
            key="simple_regression_target",
        )

        working = _regression_training_frame(df, [feature], target)
        if len(working) < 2:
            st.warning("可用樣本少於 2 筆，無法訓練線性回歸。")
            return

        _render_regression_data_intro(
            working,
            features=[feature],
            target=target,
            dataset_note="每一列是一個城市市場：x 是城市人口，y 是餐廳獲利。目標是找出一條直線來描述兩者關係。",
        )

        st.markdown("##### 訓練設定")
        c1, c2 = st.columns(2)
        learning_rate = c1.number_input(
            "學習率 α",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.4f",
            key="simple_regression_learning_rate",
        )
        epochs = c2.number_input(
            "Epoch / 迭代次數",
            min_value=1,
            max_value=5000,
            value=1500,
            step=100,
            key="simple_regression_epochs",
        )
        st.markdown("##### 模型公式")
        st.latex(r"Y = WX + B")
        _render_cost_formula()

        result_key = "simple_regression_last_artifact"
        signature = (source_label, feature, target, float(learning_rate), int(epochs), len(working))
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_simple_regression",
        )
        if not train_clicked:
            stored = st.session_state.get(result_key)
            if isinstance(stored, dict) and stored.get("signature") == signature:
                artifact = stored["artifact"]
                prediction = predict_from_artifact(artifact, working[artifact.features])
                weight = float(artifact.weights[0])
                intercept = float(artifact.intercept)
                cost = float(artifact.training_cost)
                st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
            else:
                st.info("設定 learning rate 與 epoch 後，按下「開始訓練」觀察回歸線與 Cost 的演進。")
                return
        else:
            steps = gradient_descent_steps(
                working[[feature]],
                working[target],
                learning_rate=float(learning_rate),
                epochs=int(epochs),
            )
            chart_left, chart_right = st.columns(2)
            line_placeholder = chart_left.empty()
            cost_placeholder = chart_right.empty()
            status_placeholder = st.empty()
            _animate_simple_gradient_descent(
                working,
                feature,
                target,
                steps,
                line_placeholder,
                cost_placeholder,
                status_placeholder,
            )

            final_step = steps[-1]
            prediction = predict_with_parameters(
                working[[feature]],
                final_step.weights,
                final_step.intercept,
            )
            weight = float(final_step.weights[0])
            intercept = float(final_step.intercept)
            cost = float(final_step.cost)
            artifact = LinearModelArtifact(
                model_kind="simple_linear_regression",
                features=[feature],
                target=target,
                weights=[weight],
                intercept=intercept,
                scaler=None,
                training_cost=cost,
                data_source=source_label,
            )
            st.session_state[result_key] = {"signature": signature, "artifact": artifact}

        if not train_clicked and "artifact" not in locals():
            st.info("設定 learning rate 與 epoch 後，按下「開始訓練」觀察回歸線與 Cost 的演進。")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("最後 W", f"{weight:.4f}")
        c2.metric("最後 B", f"{intercept:.4f}")
        c3.metric("最後 Cost J", f"{cost:.4f}")

        _render_prediction_error_table(working, target, prediction)
        _render_save_model_button(
            artifact=artifact,
            filename_prefix="simple_linear_regression",
        )
        _render_regression_prompts(
            [
                "請解釋這條回歸線代表什麼，並用 w 和 b 說明模型公式。",
                "請用 Cost J 說明這個模型目前預測得好不好。",
                "請找出誤差最大的幾筆資料，推測可能原因。",
            ]
        )

    _regression_page_shell(
        "單變量線性回歸",
        "使用內建餐廳獲利資料或目前 ready.csv，觀察一條回歸線如何擬合資料。",
        "內建範例資料：城市人口與餐廳獲利",
        RESTAURANT_PROFIT_PATH,
        body,
    )


def render_multiple_linear_regression_page() -> None:
    def body(df: pd.DataFrame, source_label: str) -> None:
        st.markdown("##### 多變量線性回歸")
        st.info(
            "本頁使用多個數值 features 預測一個連續 target。"
            "features 會自動做 Z-score 特徵縮放，模型 JSON 也會保存縮放參數供 inference 使用。"
        )
        numeric_columns = _numeric_regression_columns(df)
        if len(numeric_columns) < 3:
            st.warning("多變量線性回歸至少需要 2 個數值 features 與 1 個 target。")
            return

        default_features = [
            column
            for column in ["面積_平方英尺", "房間數", "樓層數", "屋齡_年"]
            if column in numeric_columns
        ]
        if not default_features:
            default_features = numeric_columns[: min(4, len(numeric_columns) - 1)]
        default_target = _default_column(numeric_columns, "房價_千美元", exclude=set(default_features))

        target = st.selectbox(
            "選擇 target（y）",
            numeric_columns,
            index=numeric_columns.index(default_target),
            key="multiple_regression_target",
        )
        feature_options = [column for column in numeric_columns if column != target]
        selected_features = st.multiselect(
            "選擇 features（x1, x2, ...）",
            feature_options,
            default=[feature for feature in default_features if feature in feature_options],
            key="multiple_regression_features",
        )
        if len(selected_features) < 2:
            st.warning("請至少選擇 2 個 features。")
            return

        working = _regression_training_frame(df, selected_features, target)
        if len(working) < 2:
            st.warning("可用樣本少於 2 筆，無法訓練線性回歸。")
            return

        try:
            scaler = create_standard_scaler(working, selected_features)
        except ValueError as exc:
            st.warning(f"無法自動縮放：{exc}")
            return
        scaled_features = apply_standard_scaler(working, scaler)

        _render_regression_data_intro(
            working,
            features=selected_features,
            target=target,
            dataset_note="每一列是一間房屋：多個 x features 共同預測房價 y。features 會先做 Z-score 縮放，再進行梯度下降。",
        )

        st.markdown("##### 訓練設定")
        c1, c2 = st.columns(2)
        learning_rate = c1.number_input(
            "學習率 α",
            min_value=0.0001,
            max_value=1.0,
            value=0.1,
            step=0.001,
            format="%.4f",
            key="multiple_regression_learning_rate",
        )
        epochs = c2.number_input(
            "Epoch / 迭代次數",
            min_value=1,
            max_value=5000,
            value=1000,
            step=100,
            key="multiple_regression_epochs",
        )

        st.markdown("##### 模型公式")
        st.latex(r"f_{\mathbf{w},b}(\mathbf{x}) = w_1x_1 + w_2x_2 + ... + b")
        _render_cost_formula()
        st.caption("此處的 w 對應 Z-score 縮放後的 features；保存模型時會一併保存 mean 與 scale。")

        result_key = "multiple_regression_last_artifact"
        signature = (
            source_label,
            tuple(selected_features),
            target,
            float(learning_rate),
            int(epochs),
            len(working),
        )
        train_clicked = st.button(
            "開始訓練",
            type="primary",
            use_container_width=True,
            key="train_multiple_regression",
        )
        if not train_clicked:
            stored = st.session_state.get(result_key)
            if isinstance(stored, dict) and stored.get("signature") == signature:
                artifact = stored["artifact"]
                prediction = predict_from_artifact(artifact, working[artifact.features])
                cost = float(artifact.training_cost)
                st.caption("顯示最近一次訓練結果；調整設定後請重新按「開始訓練」。")
            else:
                st.info("設定 learning rate 與 epoch 後，按下「開始訓練」觀察預測值與 Cost 的演進。")
                return
        else:
            steps = gradient_descent_steps(
                scaled_features,
                working[target],
                learning_rate=float(learning_rate),
                epochs=int(epochs),
            )
            chart_left, chart_right = st.columns(2)
            prediction_placeholder = chart_left.empty()
            cost_placeholder = chart_right.empty()
            status_placeholder = st.empty()
            _animate_multiple_gradient_descent(
                scaled_features,
                working[target],
                target,
                steps,
                prediction_placeholder,
                cost_placeholder,
                status_placeholder,
            )

            final_step = steps[-1]
            prediction = predict_with_parameters(
                scaled_features,
                final_step.weights,
                final_step.intercept,
            )
            cost = float(final_step.cost)
            artifact = LinearModelArtifact(
                model_kind="multiple_linear_regression",
                features=selected_features,
                target=target,
                weights=[float(value) for value in final_step.weights],
                intercept=float(final_step.intercept),
                scaler=scaler,
                training_cost=cost,
                data_source=source_label,
            )
            st.session_state[result_key] = {"signature": signature, "artifact": artifact}

        c1, c2, c3 = st.columns(3)
        c1.metric("features", f"{len(selected_features):,}")
        c2.metric("最後 B", f"{float(artifact.intercept):.4f}")
        c3.metric("Cost J", f"{cost:.4f}")
        weights = pd.DataFrame(
            {
                "feature": selected_features,
                "w": [float(value) for value in artifact.weights],
            }
        )
        st.dataframe(weights, use_container_width=True, hide_index=True)

        _render_prediction_error_table(working, target, prediction)
        _render_feature_target_overview(working, selected_features, target)
        _render_save_model_button(
            artifact=artifact,
            filename_prefix="multiple_linear_regression",
        )
        _render_regression_prompts(
            [
                "請解釋每個 w 的正負方向，以及它和 target 的關係。",
                "請說明為什麼多變量線性回歸常需要特徵縮放。",
                "請找出預測誤差最大的資料列，並說明可能原因。",
            ]
        )

    _regression_page_shell(
        "多變量線性回歸",
        "使用內建房價資料或目前 ready.csv，觀察多個 features 如何共同預測 target。",
        "內建範例資料：房價預測",
        HOUSE_PRICES_PATH,
        body,
    )


def _regression_page_shell(
    title: str,
    caption: str,
    builtin_label: str,
    builtin_path: Path,
    render_main: Callable[[pd.DataFrame, str], None],
) -> None:
    main, side = st.columns([5, 3], gap="large")
    with main:
        st.title(title)
        st.caption(caption)
        source = st.radio(
            "資料來源",
            ["內建範例資料", "目前 ready.csv"],
            horizontal=True,
            key=f"{title}_data_source",
        )
        if source == "內建範例資料":
            df = pd.read_csv(builtin_path)
            source_label = builtin_label
            st.success(
                "目前使用本頁內建教學資料。資料已整理完成，目的是專注理解演算法。"
            )
        else:
            df = load_ready_dataset()
            source_label = f"目前 ready.csv：{_display_path(READY_DATASET_PATH)}"
            if df is None:
                st.warning("尚未建立 Ready 分析就緒資料。請先建立 ready.csv，或改用內建範例資料。")
                return
            st.info(f"目前使用 `{_display_path(READY_DATASET_PATH)}`。")
        render_dataset_metrics(df)
        render_main(df, source_label)
    with side:
        render_chat_panel(
            extra_context=f"目前頁面：{title}。資料來源：{source if 'source' in locals() else '未選擇'}。",
            page_name=title,
        )


def _numeric_regression_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.select_dtypes(include="number").columns]


def _default_column(columns: list[str], preferred: str, *, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    if preferred in columns and preferred not in exclude:
        return preferred
    for column in columns:
        if column not in exclude:
            return column
    return columns[0]


def _regression_training_frame(
    df: pd.DataFrame,
    features: list[str],
    target: str,
) -> pd.DataFrame:
    columns = features + [target]
    working = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    return working


def _render_cost_formula() -> None:
    with st.expander("成本函數 J(w,b)", expanded=False):
        st.latex(r"J(w,b) = \frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})^2")
        st.caption("本頁以教材中的 Cost J 作為主要指標；Cost 越小，代表整體平方誤差越小。")


def _render_regression_data_intro(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    dataset_note: str,
) -> None:
    st.markdown("##### Data 資訊")
    st.info(dataset_note)
    role_rows = []
    for column in features + [target]:
        series = pd.to_numeric(frame[column], errors="coerce")
        role_rows.append(
            {
                "欄位": column,
                "角色": "target（y）" if column == target else "feature（x）",
                "資料型態": str(frame[column].dtype),
                "缺失值": int(frame[column].isna().sum()),
                "最小值": float(series.min()),
                "最大值": float(series.max()),
                "平均值": float(series.mean()),
            }
        )
    st.dataframe(
        pd.DataFrame(role_rows).style.format(
            {"最小值": "{:.4f}", "最大值": "{:.4f}", "平均值": "{:.4f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("資料預覽", expanded=True):
        st.dataframe(frame[features + [target]].head(10), use_container_width=True, hide_index=True)


def _animate_simple_gradient_descent(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    steps: list[GradientDescentStep],
    line_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    rendered_steps = _animation_steps(steps)
    for step in rendered_steps:
        _render_simple_step_plot(frame, feature, target, step, line_placeholder)
        _render_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，"
            f"W = {step.weights[0]:.4f}，B = {step.intercept:.4f}，Cost J = {step.cost:.4f}"
        )
        time.sleep(0.02)


def _animate_multiple_gradient_descent(
    scaled_features: pd.DataFrame,
    actual: pd.Series,
    target: str,
    steps: list[GradientDescentStep],
    prediction_placeholder,
    cost_placeholder,
    status_placeholder,
) -> None:
    rendered_steps = _animation_steps(steps)
    for step in rendered_steps:
        prediction = predict_with_parameters(scaled_features, step.weights, step.intercept)
        _render_actual_prediction_plot(actual, prediction, target, prediction_placeholder)
        _render_cost_history_plot(steps[: step.iteration + 1], cost_placeholder)
        status_placeholder.caption(
            f"Iteration {step.iteration:,} / {steps[-1].iteration:,}，"
            f"B = {step.intercept:.4f}，Cost J = {step.cost:.4f}"
        )
        time.sleep(0.02)


def _animation_steps(steps: list[GradientDescentStep]) -> list[GradientDescentStep]:
    if len(steps) <= 80:
        return steps
    stride = max(len(steps) // 80, 1)
    selected = steps[::stride]
    if selected[-1] != steps[-1]:
        selected.append(steps[-1])
    return selected


def _render_simple_step_plot(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    step: GradientDescentStep,
    placeholder,
) -> None:
    x_values = frame[feature]
    line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
    line_y = line_x * step.weights[0] + step.intercept
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.scatter(frame[feature], frame[target], alpha=0.75, label="資料點")
    ax.plot(line_x, line_y, color="red", label="回歸線")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"Y = WX + B 演進（iteration {step.iteration}）")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_cost_history_plot(steps: list[GradientDescentStep], placeholder) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot([step.iteration for step in steps], [step.cost for step in steps], color="orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J")
    ax.set_title("Cost vs Iteration")
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_actual_prediction_plot(
    actual: pd.Series,
    prediction: pd.Series,
    target: str,
    placeholder,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.scatter(actual, prediction, alpha=0.75)
    lower = float(min(actual.min(), prediction.min()))
    upper = float(max(actual.max(), prediction.max()))
    ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", label="完全預測正確")
    ax.set_xlabel(f"實際 {target}")
    ax.set_ylabel(f"預測 {target}")
    ax.set_title("實際值 vs 預測值")
    ax.legend()
    placeholder.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _render_prediction_error_table(
    frame: pd.DataFrame,
    target: str,
    prediction: pd.Series,
) -> None:
    error = prediction - frame[target]
    result = pd.DataFrame(
        {
            "actual": frame[target],
            "prediction": prediction,
            "error": error,
            "squared_error": error**2,
        }
    )
    st.markdown("##### 預測與誤差")
    st.dataframe(result.head(30).style.format("{:.4f}"), use_container_width=True)


def _render_feature_target_overview(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> None:
    with st.expander("features 與 target 關係預覽", expanded=False):
        for feature in features[:4]:
            chart_frame = frame[[feature, target]].rename(columns={feature: "feature", target: "target"})
            st.caption(f"`{feature}` vs `{target}`")
            st.scatter_chart(chart_frame, x="feature", y="target")


def _render_save_model_button(artifact: LinearModelArtifact, filename_prefix: str) -> None:
    if st.button("保存模型 JSON", type="primary", use_container_width=True, key=f"save_{filename_prefix}"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REGRESSION_MODEL_DIR / f"{filename_prefix}_{stamp}.json"
        save_model_artifact(artifact, path)
        st.success(f"已保存模型：`{_display_path(path)}`")


def _render_regression_prompts(prompts: list[str]) -> None:
    st.markdown("##### 建議問 Agent")
    for prompt in prompts:
        st.code(prompt, language="text")

