"""降維分析課程包：PCA 頁所需的整理線程式副本與課程包層 host context（ADR 0029、0030）。

本模組是 Data Science 課程包（``ui/data_science.py``、``ui/workflow_ui.py``）的獨立副本，
不引用 Data Science 課程包的檔案；只裝 pca 時也能運作，沒有 Working 時只在頁上提示。
修 bug 時請一併檢查 Data Science 課程包的原版。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ui.data_ui import (
    WORKSPACE_DIR,
    display_path,
    render_chat_panel,
)
from dataset_streamlit_shell.ui.dual_pane_shell import open_content_dual_pane

ORIGINAL_DATASET_PATH = WORKSPACE_DIR / "original.csv"
WORKING_DATASET_PATH = WORKSPACE_DIR / "working.csv"
READY_DATASET_PATH = WORKSPACE_DIR / "ready.csv"
TRAIN_DATASET_PATH = WORKSPACE_DIR / "train.csv"
VAL_DATASET_PATH = WORKSPACE_DIR / "val.csv"
TEST_DATASET_PATH = WORKSPACE_DIR / "test.csv"
CLEANING_LOG_PATH = WORKSPACE_DIR / "cleaning_log.jsonl"
DUAL_TABLE_COPIES_DIR = WORKSPACE_DIR / "integration"
SPLIT_DATASET_PATHS = (TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH)

MISSING_WORKING_HINT = "尚未建立 Working 工作資料。請先加裝 Data Science 課程包並建立 Working 工作資料。"


# ---- Working 讀取（Data Science 課程包副本） ----


def clear_split_datasets() -> None:
    for path in SPLIT_DATASET_PATHS:
        if path.exists():
            path.unlink()


def discard_leftover_ready() -> None:
    st.session_state.pop("ready_dataset_df", None)
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


def load_working_dataset() -> pd.DataFrame | None:
    discard_leftover_ready()
    sync_split_if_working_stale()
    if not WORKING_DATASET_PATH.exists():
        return None
    df = pd.read_csv(WORKING_DATASET_PATH)
    st.session_state["working_dataset_df"] = df
    return df


def pca_missing_working_hint(working_path: Path | None = None) -> str | None:
    """PCA 頁的前置狀態：沒有 Working 時回傳提示（不自動加裝 Data Science），有則 None。"""
    path = WORKING_DATASET_PATH if working_path is None else working_path
    return None if path.exists() else MISSING_WORKING_HINT


def pca_status(df: pd.DataFrame) -> dict[str, object]:
    numeric = df.select_dtypes(include="number")
    return {
        "rows": len(df),
        "numeric_columns": len(numeric.columns),
        "missing_cells": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        "enough_columns": len(numeric.columns) >= 2,
        "enough_rows": len(df) >= 3,
    }


# ---- host context ----


def pca_pack_layer() -> str:
    """降維分析課程包層：Data Science 課程包層的副本（Working 規則與可寫清單）。"""
    source = display_path(ORIGINAL_DATASET_PATH)
    working = display_path(WORKING_DATASET_PATH)
    cleaning_log = display_path(CLEANING_LOG_PATH)
    passengers_copy = display_path(DUAL_TABLE_COPIES_DIR / "passengers.csv")
    voyage_copy = display_path(DUAL_TABLE_COPIES_DIR / "voyage.csv")
    return (
        "【降維分析課程包】本頁屬於降維分析，以 Working 工作資料做 PCA。"
        f"可寫清單：{working}；{cleaning_log}；"
        f"雙表工作副本 {passengers_copy}、{voyage_copy}；"
        f"資料整合合併時可與 Working 同批寫入 {source}。"
        f"Original 原始資料路徑：{source}，只作為重置來源，請勿覆蓋。"
        f"Working 工作資料路徑：{working}。"
        "圖表探索、PCA、資料切分都讀 Working。"
        "寫回 Working 會作廢 train／val／test，並刪掉遺留 ready.csv。"
        "不要建立 ready.csv，也不要把它當分析來源。"
        "Working 不存在時，請提醒學生先加裝 Data Science 課程包，"
        "到「欄位與資料整合」建立 Working 工作資料；不要自行建立 Working。"
        f"如果使用者要求補值、清理資料、計算欄位或新增欄位，且 Working 已存在，請預設讀取並更新 {working}，不要覆蓋 {source}。"
        f"修改 Working 工作資料後，請追加一筆 JSONL 紀錄到 {cleaning_log}，每行必須是單一 JSON 物件，"
        "欄位固定為 created_at、actor、action、columns、rows、note。"
        "actor 必須是 agent；action 使用簡短 snake_case；columns 是受影響欄位陣列；"
        "rows 是受影響筆數；note 用繁體中文一句話摘要修改內容。"
    )


def pca_host_context() -> str:
    """PCA 頁的 host：框架層 + 降維分析課程包層。"""
    return assemble_host_context(pca_pack_layer())


def pca_page_snapshot(extra_context: str = "") -> str:
    """每輪【目前頁面狀態】：Working 摘要（或尚未建立的提示）＋該頁補充。"""
    df = load_working_dataset()
    parts: list[str] = []
    if df is None:
        parts.append(MISSING_WORKING_HINT + "一般 PCA 概念問題可直接回答。")
    else:
        columns = ", ".join(str(c) for c in df.columns)
        parts.append(f"工作資料目前有 {len(df)} 筆、{len(df.columns)} 欄。欄位：{columns}。")
    if extra_context.strip():
        parts.append(extra_context.strip())
    return "\n".join(parts)


PCA_GREETING = (
    "請先按「啟用資料 Agent」。啟用後，我可以協助你理解 PCA 主成分分析，"
    "並一起解讀 Working 工作資料的主成分。"
)


def render_pca_chat(extra_context: str = "", page_name: str = "") -> None:
    """PCA 頁的聊天欄：降維分析 host，每輪送出時才讀 Working 摘要。"""
    render_chat_panel(
        extra_context=lambda: pca_page_snapshot(extra_context),
        page_name=page_name,
        host_context=pca_host_context(),
        greeting=PCA_GREETING,
    )


def render_pca_shell(
    title: str,
    caption: str,
    render_main: Callable[[pd.DataFrame], None],
) -> None:
    """PCA 頁雙欄殼：有 Working 時交給 render_main，沒有時只顯示提示。"""
    teaching, agent = open_content_dual_pane()
    with teaching:
        st.title(title)
        st.caption(caption)
        hint = pca_missing_working_hint()
        df = None if hint else load_working_dataset()
        if df is None:
            st.warning(hint or MISSING_WORKING_HINT)
        else:
            render_main(df)
    with agent:
        render_pca_chat(page_name=title)
