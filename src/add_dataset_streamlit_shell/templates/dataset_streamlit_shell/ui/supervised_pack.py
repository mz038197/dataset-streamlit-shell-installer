"""監督式課程包層 host context（ADR 0030）。

線性回歸、邏輯迴歸、線性支持向量機、K-近鄰分類、決策樹與集成五頁共用；
有寫回協定的頁把協定當頁面層傳入 ``supervised_host_context``。
"""

from __future__ import annotations

from dataset_streamlit_shell.host_context import SHELL_ROOT, assemble_host_context

_BUILTIN = f"{SHELL_ROOT.name}/built-in-data"


def supervised_pack_layer() -> str:
    """監督式課程包層：只用內建範例資料；本層沒有可寫檔，可寫檔只由頁面層列出。"""
    return (
        "【監督式課程包】本頁屬於機器學習 · 監督式，使用內建範例資料教學："
        f"{_BUILTIN}/regression 與 {_BUILTIN}/classification 底下的 CSV。"
        "內建範例資料只讀，不得覆寫、刪除或另存成新檔。"
        "回答本頁問題時以頁上載入的內建範例資料為準，不要改用專案中其他資料檔。"
        "本層可寫清單：無；若本頁另有寫回協定，只寫該協定列出的檔案。"
    )


def supervised_host_context(page_layer: str | None = None) -> str:
    """監督式各頁的 host：框架層 + 監督式課程包層（+ 該頁寫回協定）。"""
    return assemble_host_context(supervised_pack_layer(), page_layer)
