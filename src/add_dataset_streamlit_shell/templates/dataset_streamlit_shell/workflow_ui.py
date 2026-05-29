from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dataset_streamlit_shell.data_ui import (
    CLEANING_LOG_PATH,
    READY_DATASET_PATH,
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


PromptList = list[str]
CATEGORICAL_SELECTION_STATE_KEY = "confirmed_categorical_columns"
CATEGORICAL_SELECTION_WIDGET_KEY = "selected_categorical_columns_widget"
CATEGORICAL_SELECTION_EDIT_WIDGET_KEY = "selected_categorical_columns_edit_widget"
CORRELATION_SELECTION_STATE_KEY = "confirmed_correlation_columns"
CORRELATION_SELECTION_WIDGET_KEY = "selected_correlation_columns_widget"


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
        render_chat_panel(extra_context=extra_context)


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


def _teaching_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and _column_kind(df[column]) == "數值"
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

    _page_shell("刪除重複資料列", "讓學生先定義重複規則，再請 Agent 刪除重複列。", body)


def render_outliers_page() -> None:
    def body(df: pd.DataFrame) -> None:
        numeric_columns = _teaching_numeric_columns(df)
        numeric = df[numeric_columns]
        st.markdown("##### 診斷：離群值")
        if numeric.empty:
            st.warning("目前沒有適合檢查離群值的連續數值欄位。")
            return

        st.caption(
            "離群值不是固定答案。請先選擇判斷方法，系統才會依照該方法列出可能有離群值的欄位。"
        )
        method = st.selectbox(
            "離群值判斷方法",
            ["請選擇方法", "IQR 法", "Z-score 法"],
            key="outlier_method",
        )
        if method == "請選擇方法":
            st.info("你還沒有定義什麼算離群值，因此系統不會先判斷。")
            return

        if method == "IQR 法":
            outlier_frame = _iqr_outlier_summary(df, numeric)
        else:
            threshold = st.slider("Z-score 閾值", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
            outlier_frame = _zscore_outlier_summary(df, numeric, threshold)

        if outlier_frame.empty:
            st.warning("目前數值欄位沒有足夠資料可檢查離群值。")
            return

        st.markdown("##### 有離群值的欄位")
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

    _page_shell("離群值檢查", "專心檢查極端數值，避免和缺失值、分布探索混在一起。", body)


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


def render_correlation_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 診斷：數值相關性")
        st.info(
            "請先和 Agent 討論哪些欄位適合做數值相關性分析，再在下方選取欄位。"
            "選取至少兩個欄位後，系統只顯示這些欄位之間的完整相關矩陣。"
        )
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
                "請說明這些欄位的相關性對後續 PCA 分析可能有什麼影響。",
            ]
        )

    _page_shell(
        "數值相關性",
        "在建立 Ready 分析就緒資料之前，檢查學生選取欄位之間的數值關係。",
        body,
        extra_context_builder=_correlation_extra_context,
    )


def render_encoding_correlation_page() -> None:
    render_encoding_page()


def render_ready_page() -> None:
    def body(df: pd.DataFrame) -> None:
        st.markdown("##### 建立 Ready 分析就緒資料")
        st.caption("將目前 Working 工作資料凍結為穩定的 `ready.csv`，供 Wald / PCA / K-Means 使用。")
        missing_total = int(df.isna().sum().sum())
        object_cols = len(df.select_dtypes(include=["object", "string", "category"]).columns)
        duplicate_rows = int(df.duplicated().sum())
        numeric_cols = len(df.select_dtypes(include="number").columns)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("列數", f"{len(df):,}")
        c2.metric("數值欄位", f"{numeric_cols:,}")
        c3.metric("缺失儲存格", f"{missing_total:,}")
        c4.metric("重複列", f"{duplicate_rows:,}")
        if object_cols:
            st.warning(f"仍有 {object_cols} 個文字/類別欄位。PCA 可能需要先做編碼或只選數值欄位。")
        if missing_total:
            st.warning("仍有缺失值。Wald / PCA 前建議先完成缺失值處理。")
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


def wald_status(df: pd.DataFrame) -> dict[str, object]:
    numeric = df.select_dtypes(include="number")
    binary_columns = [
        str(column)
        for column in df.columns
        if df[column].dropna().nunique() == 2
    ]
    return {
        "rows": len(df),
        "numeric_columns": len(numeric.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "binary_columns": binary_columns,
    }


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
            st.warning("尚未建立 Ready 分析就緒資料。請先到「建立分析資料集」頁完成匯出。")
            return
        render_main(df)
    with side:
        render_chat_panel()

