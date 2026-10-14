"""深度學習課程包層 host context（ADR 0030）：內建範例資料邊界與可寫白名單。"""

from __future__ import annotations

from dataset_streamlit_shell.host_context import assemble_host_context
from dataset_streamlit_shell.ui.data_ui import SHELL_ROOT, WORKSPACE_DIR, _display_path
from dataset_streamlit_shell.ui.nn_form_state import (
    nn_form_path,
    nn_last_run_path,
    nn_train_request_path,
)

BUILTIN_NN_DATA_PATH = SHELL_ROOT / "built-in-data" / "classification" / "nn_binary_400.csv"


def deep_learning_pack_layer() -> str:
    """深度學習課程包層：只用內建範例資料；寫檔只限本包可寫清單。"""
    builtin = _display_path(BUILTIN_NN_DATA_PATH)
    form = _display_path(nn_form_path(WORKSPACE_DIR))
    request = _display_path(nn_train_request_path(WORKSPACE_DIR))
    last_run = _display_path(nn_last_run_path(WORKSPACE_DIR))
    return (
        "【深度學習課程包】本頁屬於深度學習，使用內建範例資料 "
        f"{builtin}，只讀不寫，也不要改用其他資料檔。"
        f"可寫清單：{form}；{request}。"
        f"{last_run} 由系統在訓練結束後寫入，只讀不寫。"
    )


def deep_learning_host_context(page_layer: str | None = None) -> str:
    """深度學習各頁的 host：框架層 + 深度學習課程包層（+ 可選的頁面層）。"""
    return assemble_host_context(deep_learning_pack_layer(), page_layer)
