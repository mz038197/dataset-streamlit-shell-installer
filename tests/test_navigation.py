from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.navigation import NavPage, installed_sections  # noqa: E402


def _overview() -> None:
    pass


SECTIONS = {
    "": [NavPage(_overview, "總覽", default=True)],
    "AI 協作資料整理": [
        NavPage("pages/3_Field_Quality.py", "欄位與資料整合"),
        NavPage("pages/17_Data_Transform.py", "資料轉換"),
        NavPage("pages/4_Duplicates.py", "刪除重複資料列"),
    ],
    "機器學習 · 監督式": [
        NavPage("pages/14_Simple_Linear_Regression.py", "線性回歸"),
    ],
    "降維分析": [
        NavPage("pages/10_PCA.py", "PCA 主成分分析"),
    ],
}


def _make_shell(root: Path, page_files: list[str]) -> Path:
    for rel in page_files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return root


def test_partial_shell_keeps_only_existing_pages_in_order(tmp_path: Path) -> None:
    shell = _make_shell(
        tmp_path,
        ["pages/4_Duplicates.py", "pages/3_Field_Quality.py", "pages/10_PCA.py"],
    )

    result = installed_sections(shell, SECTIONS)

    assert list(result) == ["", "AI 協作資料整理", "降維分析"]
    assert [p.title for p in result["AI 協作資料整理"]] == ["欄位與資料整合", "刪除重複資料列"]
    assert [p.title for p in result["降維分析"]] == ["PCA 主成分分析"]


def test_overview_kept_even_when_no_page_files_exist(tmp_path: Path) -> None:
    result = installed_sections(tmp_path, SECTIONS)

    assert list(result) == [""]
    assert result[""][0].title == "總覽"
    assert result[""][0].default is True


def test_all_pages_present_returns_full_definition(tmp_path: Path) -> None:
    shell = _make_shell(
        tmp_path,
        [
            "pages/3_Field_Quality.py",
            "pages/17_Data_Transform.py",
            "pages/4_Duplicates.py",
            "pages/14_Simple_Linear_Regression.py",
            "pages/10_PCA.py",
        ],
    )

    assert installed_sections(shell, SECTIONS) == SECTIONS


def test_navigation_module_does_not_import_streamlit(tmp_path: Path) -> None:
    import subprocess

    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import dataset_streamlit_shell.navigation;"
        "assert 'streamlit' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(TEMPLATE_ROOT)], check=True)
