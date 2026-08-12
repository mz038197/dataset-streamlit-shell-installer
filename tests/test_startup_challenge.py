from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
SHELL = TEMPLATE_ROOT / "dataset_streamlit_shell"
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui.startup_challenge_context import (  # noqa: E402
    CHALLENGE_COMPANIES,
    challenge_host_context,
    challenge_page_snapshot,
    challenge_paths,
    clear_challenge_working,
    company_changed_should_clear_working,
)

APP = SHELL / "app.py"
PAGES = SHELL / "pages"
UI = SHELL / "ui"


def test_challenge_companies_are_five() -> None:
    assert CHALLENGE_COMPANIES == (
        "edupulse",
        "vitalrisk",
        "airsense",
        "churnlab",
        "flowcast",
    )


def test_challenge_paths_point_under_challenge_dir(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    assert paths.start_csv == tmp_path / "challenge" / "edupulse.csv"
    assert paths.handbook == tmp_path / "challenge" / "edupulse_資料說明書.md"
    assert paths.working_csv == tmp_path / "challenge" / "working.csv"


def test_host_context_is_standalone_and_includes_company_fragment() -> None:
    text = challenge_host_context(
        company="edupulse",
        start_csv="dataset_streamlit_shell/workspace/challenge/edupulse.csv",
        handbook="dataset_streamlit_shell/workspace/challenge/edupulse_資料說明書.md",
        working_csv="dataset_streamlit_shell/workspace/challenge/working.csv",
        scripts_dir="dataset_streamlit_shell/scripts",
    )
    assert "【AI Startup Challenge 模式】" in text
    assert "白板三塊" in text or "我們在解決什麼" in text
    assert "edupulse" in text
    assert "G1" in text or "G2" in text
    assert "nn_form.json" not in text
    assert "Challenge 工作資料" in text
    # Must not push Titanic Original／Working 語意當挑戰主軌道。
    assert "只作為重置來源" not in text
    assert "不要改根目錄的 original.csv" in text


def test_host_context_vitalrisk_fragment() -> None:
    text = challenge_host_context(
        company="vitalrisk",
        start_csv="challenge/vitalrisk.csv",
        handbook="challenge/vitalrisk_資料說明書.md",
        working_csv="challenge/working.csv",
        scripts_dir="scripts",
    )
    assert "假陽性" in text or "假陰性" in text
    assert "不是診斷" in text


def test_clear_challenge_working_removes_file(tmp_path: Path) -> None:
    challenge_dir = tmp_path / "challenge"
    challenge_dir.mkdir()
    working = challenge_dir / "working.csv"
    working.write_text("a,b\n1,2\n", encoding="utf-8")
    assert clear_challenge_working(working) is True
    assert not working.exists()
    assert clear_challenge_working(working) is False


def test_company_change_clears_when_previous_differs() -> None:
    assert company_changed_should_clear_working(None, "edupulse") is False
    assert company_changed_should_clear_working("edupulse", "edupulse") is False
    assert company_changed_should_clear_working("edupulse", "vitalrisk") is True


def test_page_snapshot_mentions_files_and_board() -> None:
    snap = challenge_page_snapshot(
        company="edupulse",
        start_exists=True,
        working_exists=False,
        board_summary="①空殼 ②TODO ③空殼",
        target_column="",
    )
    assert "edupulse" in snap
    assert "Challenge 起點資料" in snap
    assert "Challenge 工作資料" in snap
    assert "不存在" in snap
    assert "①空殼" in snap


def test_app_nav_has_workshop_section_and_page() -> None:
    src = APP.read_text(encoding="utf-8")
    assert '"AI新創工作坊"' in src
    assert "專案展示" in src
    assert "30_Startup_Challenge.py" in src
    dim = src.index('"降維分析"')
    workshop = src.index('"AI新創工作坊"')
    assert workshop > dim


def test_page_and_ui_files_exist() -> None:
    assert (PAGES / "30_Startup_Challenge.py").is_file()
    assert (UI / "startup_challenge_ui.py").is_file()
    assert (UI / "startup_challenge_context.py").is_file()


def test_empty_shell_marks_todo_and_ai_coding_board() -> None:
    ui = (UI / "startup_challenge_ui.py").read_text(encoding="utf-8")
    assert "TODO(challenge)" in ui
    assert "BOARD_CUSTOMER" in ui
    assert "challenge_host_context" in ui
    assert "agent_scope" in ui or 'host_context=' in ui
