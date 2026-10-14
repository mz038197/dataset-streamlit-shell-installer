"""AI Lab 框架的 host context：框架層，以及唯一的三層組裝方式（ADR 0030）。"""

from __future__ import annotations

from pathlib import Path

SHELL_ROOT = Path(__file__).resolve().parent


def framework_host_context() -> str:
    """框架層：角色、先讀檔、腳本位置、白名單寫檔。不含任何課程包的領域規則。"""
    scripts = f"{SHELL_ROOT.name}/scripts"
    return (
        "你是 AI Lab 的資料 Agent，協助學生在本專案中學習資料分析與 AI。"
        "回答資料問題前，請使用你的 read_file 或 exec 工具實際讀取檔案後再回答，不要憑空推測內容。"
        f"如果需要撰寫 Python 腳本，請只建立在 {scripts} 底下，"
        "不要在專案根目錄建立臨時 Python 腳本。"
        "寫檔只限各層明確列出的可寫清單；沒有列在可寫清單的檔案一律不寫、不覆蓋、不刪除，"
        "學生要求時請說明這不在本頁的可寫範圍。"
    )


def assemble_host_context(
    pack_layer: str | None = None,
    page_layer: str | None = None,
) -> str:
    """依「框架層 → 課程包層 → 頁面層」組出完整 host context。

    頁面層只能疊在課程包層之上：沒有課程包層時，傳入的頁面層會被丟掉，只剩框架層
    （避免某頁繞過課程包的資料邊界與可寫清單）。
    """
    layers = [framework_host_context()]
    if pack_layer:
        layers.append(pack_layer)
        if page_layer:
            layers.append(page_layer)
    return "\n\n".join(layers)
