"""非監督式課程包：課程包層 host context 與分群頁的聊天欄（ADR 0030）。

分群頁只用內建範例資料；本層沒有可寫的根目錄檔。
"""

from __future__ import annotations

from dataset_streamlit_shell.host_context import SHELL_ROOT, assemble_host_context
from dataset_streamlit_shell.ui.data_ui import display_path, render_chat_panel

CLUSTERING_EXAMPLES_DIR = SHELL_ROOT / "built-in-data" / "clustering"


def unsupervised_pack_layer() -> str:
    """非監督式課程包層：內建範例邊界，可寫清單為空（白名單）。"""
    examples = display_path(CLUSTERING_EXAMPLES_DIR)
    return (
        "【非監督式課程包】本頁屬於機器學習 · 非監督式（分群）。"
        f"本頁只使用內建範例資料 {examples}，只讀不寫，也不要改用專案中其他資料檔。"
        "可寫清單：無；本層沒有任何可寫的檔案。"
        "分群演算法不使用教學用真實群標籤；該欄僅供對照。"
    )


def unsupervised_host_context() -> str:
    """分群頁的 host：框架層 + 非監督式課程包層。"""
    return assemble_host_context(unsupervised_pack_layer())


def render_unsupervised_chat(extra_context: str = "", page_name: str = "") -> None:
    """分群頁的聊天欄：非監督式 host，每輪附上該頁狀態。"""
    render_chat_panel(
        extra_context=extra_context,
        page_name=page_name,
        host_context=unsupervised_host_context(),
    )
