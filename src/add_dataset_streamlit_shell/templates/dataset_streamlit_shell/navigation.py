"""側欄分段過濾：只留下殼層中實際存在的頁（不依賴 Streamlit）。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NavPage:
    """一個側欄項目：source 為相對殼層根目錄的頁面路徑，或非檔案頁（例如總覽）的函式。"""

    source: str | Callable[[], None]
    title: str
    default: bool = False


def installed_sections(
    shell_root: Path, sections: Mapping[str, Sequence[NavPage]]
) -> dict[str, list[NavPage]]:
    """回傳只含存在頁面的分段；空分段剔除，分段與頁面順序不變。非檔案頁永遠保留。"""
    result: dict[str, list[NavPage]] = {}
    for section, pages in sections.items():
        kept = [
            page
            for page in pages
            if not isinstance(page.source, str) or (shell_root / page.source).is_file()
        ]
        if kept:
            result[section] = kept
    return result


def needs_manual_menu(sections: Mapping[str, Sequence[NavPage]]) -> bool:
    """只剩一頁時 st.navigation 會自動隱藏側欄選單，須由 app.py 自己畫出。"""
    return sum(len(pages) for pages in sections.values()) == 1
