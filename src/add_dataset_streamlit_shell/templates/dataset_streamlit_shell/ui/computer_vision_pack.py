"""電腦視覺課程包：課程包層 host context 與聊天欄（ADR 0030）。

CNN 入門與五個電腦視覺分頁都經 ``render_computer_vision_chat`` 開聊天欄，
明確傳入「框架層 + 電腦視覺課程包層」。
"""

from __future__ import annotations

import streamlit as st

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ui.data_ui import SHELL_ROOT, _display_path, render_chat_panel

CV_DATA_DIR = SHELL_ROOT / "built-in-data" / "computer-vision"


def computer_vision_pack_layer() -> str:
    """電腦視覺課程包層：只用內建範例圖片與頁面上的模型結果；可寫清單為空（白名單）。"""
    examples = _display_path(CV_DATA_DIR / "examples")
    models = _display_path(CV_DATA_DIR / "models")
    return (
        "【電腦視覺課程包】本頁屬於電腦視覺（CNN、影像分類、物件偵測、語意分割、實例分割、提示式分割）。"
        f"本頁使用內建範例圖片 {examples} 與學生在頁面上傳的圖片；"
        "模型推論與訓練由頁面按鈕執行，請根據【目前頁面狀態】中的結果解釋，不要自行重跑模型。"
        f"模型權重 {models} 由頁面下載與管理，只讀不改。"
        "可寫清單：無。本層沒有可寫的檔案；內建範例圖片與模型權重不得覆寫或刪除。"
    )


def computer_vision_host_context() -> str:
    return assemble_host_context(computer_vision_pack_layer())


def render_computer_vision_chat(*, page_title: str, context_key: str) -> None:
    """電腦視覺各頁的聊天欄：CV host，頁面狀態取自 ``st.session_state[context_key]``。"""
    render_chat_panel(
        extra_context=str(st.session_state.get(context_key, f"目前頁面：{page_title}。")),
        page_name=page_title,
        host_context=computer_vision_host_context(),
    )
