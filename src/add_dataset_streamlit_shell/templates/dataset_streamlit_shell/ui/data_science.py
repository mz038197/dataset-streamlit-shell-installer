"""Data Science 課程包：Original／Working／雙表／切分等工作檔，與課程包層 host context。

AI Lab 框架不知道這些概念；AI 協作資料整理各頁經 ``render_data_science_chat``
開聊天欄，明確傳入「框架層 + Data Science 課程包層」，每輪附上 Working 摘要。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

import pandas as pd
import streamlit as st

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ml.integration import clear_dual_table_copies
from dataset_streamlit_shell.ui.data_ui import (
    WORKSPACE_DIR,
    _display_path,
    render_chat_panel,
)

ORIGINAL_DATASET_PATH = WORKSPACE_DIR / "original.csv"
WORKING_DATASET_PATH = WORKSPACE_DIR / "working.csv"
READY_DATASET_PATH = WORKSPACE_DIR / "ready.csv"
TRAIN_DATASET_PATH = WORKSPACE_DIR / "train.csv"
VAL_DATASET_PATH = WORKSPACE_DIR / "val.csv"
TEST_DATASET_PATH = WORKSPACE_DIR / "test.csv"
CLEANING_LOG_PATH = WORKSPACE_DIR / "cleaning_log.jsonl"
DUAL_TABLE_COPIES_DIR = WORKSPACE_DIR / "integration"
SPLIT_DATASET_PATHS = (TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH)


def _ensure_workspace_dir() -> None:
    WORKSPACE_DIR.mkdir(exist_ok=True)


def save_dataset(df: pd.DataFrame, *, working: bool = False) -> None:
    _ensure_workspace_dir()
    target = WORKING_DATASET_PATH if working else ORIGINAL_DATASET_PATH
    df.to_csv(target, index=False)
    if working:
        discard_leftover_ready()
        clear_split_datasets()
        return
    st.session_state["dataset_df"] = df
    st.session_state.pop("working_dataset_df", None)
    st.session_state.pop("selected_columns", None)
    st.session_state.pop("filter_columns", None)
    st.session_state.pop("filter_signature", None)


def refresh_working_dataset_cache() -> None:
    st.session_state.pop("working_dataset_df", None)


def refresh_ready_dataset_cache() -> None:
    st.session_state.pop("ready_dataset_df", None)


def load_dataset() -> pd.DataFrame | None:
    if "dataset_df" in st.session_state:
        return st.session_state["dataset_df"]
    if ORIGINAL_DATASET_PATH.exists():
        df = pd.read_csv(ORIGINAL_DATASET_PATH)
        st.session_state["dataset_df"] = df
        return df
    return None


def load_working_dataset() -> pd.DataFrame | None:
    discard_leftover_ready()
    sync_split_if_working_stale()
    if WORKING_DATASET_PATH.exists():
        df = pd.read_csv(WORKING_DATASET_PATH)
        st.session_state["working_dataset_df"] = df
        return df
    return load_dataset()


def reset_working_dataset() -> None:
    st.session_state.pop("working_dataset_df", None)
    if WORKING_DATASET_PATH.exists():
        WORKING_DATASET_PATH.unlink()
    discard_leftover_ready()
    clear_split_datasets()
    reset_cleaning_log()


def clear_split_datasets() -> None:
    for path in SPLIT_DATASET_PATHS:
        if path.exists():
            path.unlink()


def discard_leftover_ready() -> None:
    refresh_ready_dataset_cache()
    if READY_DATASET_PATH.exists():
        READY_DATASET_PATH.unlink()


def sync_split_if_working_stale() -> bool:
    if not WORKING_DATASET_PATH.exists():
        return False
    split_files = [path for path in SPLIT_DATASET_PATHS if path.exists()]
    if not split_files:
        return False
    working_mtime = WORKING_DATASET_PATH.stat().st_mtime
    if not any(working_mtime > path.stat().st_mtime for path in split_files):
        return False
    clear_split_datasets()
    return True


def working_dataset_file_exists() -> bool:
    return WORKING_DATASET_PATH.exists()


def clear_to_dual_start() -> None:
    """清除 working／切分產物／original／雙表工作副本與遺留 ready.csv，回到雙表起點。"""
    clear_split_datasets()
    reset_working_dataset()
    st.session_state.pop("dataset_df", None)
    if ORIGINAL_DATASET_PATH.exists():
        ORIGINAL_DATASET_PATH.unlink()
    clear_dual_table_copies(WORKSPACE_DIR)


def reset_cleaning_log() -> None:
    if CLEANING_LOG_PATH.exists():
        CLEANING_LOG_PATH.unlink()


def initialize_working_dataset(df: pd.DataFrame) -> None:
    save_dataset(df, working=True)
    st.session_state["working_dataset_df"] = df


def reset_working_dataset_from_source() -> bool:
    source = load_dataset()
    if source is None:
        return False
    save_dataset(source, working=True)
    st.session_state["working_dataset_df"] = source
    append_cleaning_log(
        action="重置工作資料",
        columns=source.columns,
        rows=len(source),
        note="使用 original.csv 重建 working.csv。",
        actor="ui",
    )
    return True


def save_split_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    _ensure_workspace_dir()
    train.to_csv(TRAIN_DATASET_PATH, index=False)
    val.to_csv(VAL_DATASET_PATH, index=False)
    test.to_csv(TEST_DATASET_PATH, index=False)


def append_cleaning_log(
    *,
    action: str,
    columns: Iterable[str] | None = None,
    rows: int | None = None,
    note: str = "",
    actor: str = "ui",
) -> None:
    _ensure_workspace_dir()
    column_list = [] if columns is None else [str(column) for column in columns]
    normalized_actor = actor if actor in {"agent", "ui"} else "ui"
    entry = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "actor": normalized_actor,
        "action": action,
        "columns": column_list,
        "rows": rows,
        "note": note,
    }
    with CLEANING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_cleaning_log(limit: int = 8) -> list[dict[str, object]]:
    if not CLEANING_LOG_PATH.exists():
        return []
    entries: list[dict[str, object]] = []
    with CLEANING_LOG_PATH.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries[-limit:][::-1]


def data_science_pack_layer() -> str:
    """Data Science 課程包層：雙表、Original、Working 與 cleaning_log 的整理規則與可寫清單。"""
    source = _display_path(ORIGINAL_DATASET_PATH)
    working = _display_path(WORKING_DATASET_PATH)
    cleaning_log = _display_path(CLEANING_LOG_PATH)
    passengers_copy = _display_path(DUAL_TABLE_COPIES_DIR / "passengers.csv")
    voyage_copy = _display_path(DUAL_TABLE_COPIES_DIR / "voyage.csv")
    return (
        "【Data Science 課程包】本頁屬於 AI 協作資料整理。"
        f"可寫清單：{working}；{cleaning_log}；"
        f"雙表工作副本 {passengers_copy}、{voyage_copy}；"
        f"資料整合合併時可與 Working 同批寫入 {source}。"
        f"Original 原始資料路徑：{source}，只作為重置來源。"
        "合併完成前不存在；僅在資料整合時可與 Working 同批寫入，此外請勿覆蓋。"
        f"Working 工作資料路徑：{working}。"
        "資料整合成功時與 Original 一併寫入；Working 為之後整理用副本。"
        "圖表探索、PCA、資料切分都讀 Working。"
        "寫回 Working 會作廢 train／val／test，並刪掉遺留 ready.csv。"
        "不要建立 ready.csv，也不要把它當分析來源。"
        "雙表起點進頁時已寫出這兩份工作副本；請直接改航程表副本的 passenger_id 為 PassengerId，不要另存新檔。"
        "內建雙表教材不得覆寫。"
        "合併前只准將航程表副本的 passenger_id 改名為 PassengerId；不要補值、刪欄或改乘客表。"
        "通過訓練前預測（資料整合）且鍵名已對齊後，才可依學生指定的 left 或 inner 合併"
        "（課堂主線建議 left），"
        f"同時寫入 {source} 與 {working}，並刪除雙表工作副本"
        "（可用 commit_dual_table_merge）。"
        "寫入後請告訴學生按「重新讀取工作資料」。"
        "未解鎖或未對齊鍵名時拒絕合併。"
        f"如果使用者要求補值、清理資料、計算欄位或新增欄位，且 Working 已存在，請預設讀取並更新 {working}，不要覆蓋 {source}。"
        f"修改 Working 工作資料後，請追加一筆 JSONL 紀錄到 {cleaning_log}，每行必須是單一 JSON 物件，"
        "欄位固定為 created_at、actor、action、columns、rows、note。"
        "actor 必須是 agent；action 使用簡短 snake_case；columns 是受影響欄位陣列；"
        "rows 是受影響筆數；note 用繁體中文一句話摘要修改內容。"
        "範例："
        '{"created_at":"2026-05-29T12:05:41","actor":"agent",'
        '"action":"fill_missing_age","columns":["Age"],"rows":177,'
        '"note":"以中位數補齊 Age 欄位的空值。"}'
    )


def data_science_host_context() -> str:
    """AI 協作資料整理各頁的 host：框架層 + Data Science 課程包層。"""
    return assemble_host_context(data_science_pack_layer())


def working_page_snapshot(extra_context: str = "") -> str:
    """每輪【目前頁面狀態】：Working 摘要（或尚未建立的提示）＋該頁補充。"""
    df = load_working_dataset()
    parts: list[str] = []
    if df is None:
        parts.append(
            "尚未建立工作資料。"
            "若詢問資料內容、清理、補值或欄位計算，請先提醒到「欄位與資料整合」查看雙表；"
            "合併前可請 Agent 對齊航程表鍵名，通過關卡後再請 Agent 合併，會同時寫入 Original 與 Working。"
            "一般概念問題可直接回答。"
        )
    else:
        columns = ", ".join(str(c) for c in df.columns)
        parts.append(f"工作資料目前有 {len(df)} 筆、{len(df.columns)} 欄。欄位：{columns}。")
    if extra_context.strip():
        parts.append(extra_context.strip())
    return "\n".join(parts)


DATA_SCIENCE_GREETING = (
    "請先按「啟用資料 Agent」。啟用後，我可以協助你理解資料整理流程；"
    "雙表合併建立工作資料後，也能一起分析 Working。"
)


def render_data_science_chat(extra_context: str = "", page_name: str = "") -> None:
    """AI 協作資料整理各頁的聊天欄：DS host，每輪送出時才讀 Working 摘要。"""
    render_chat_panel(
        extra_context=lambda: working_page_snapshot(extra_context),
        page_name=page_name,
        host_context=data_science_host_context(),
        greeting=DATA_SCIENCE_GREETING,
    )
