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
    DATA_VIEW_START,
    DATA_VIEW_TEST,
    DATA_VIEW_TRAIN,
    DATA_VIEW_WORKING,
    available_data_views,
    challenge_host_context,
    challenge_page_snapshot,
    challenge_paths,
    clear_challenge_runtime,
    company_changed_should_reset,
    default_data_view,
    artifact_matches_current_split,
    invalidate_challenge_split,
    model_zone_unlocked,
    restore_startup_challenge_ui,
    result_zone_unlocked,
    split_files_ready,
    split_signature,
    sync_split_if_working_stale,
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


def test_challenge_paths_include_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    assert paths.start_csv == tmp_path / "challenge" / "edupulse.csv"
    assert paths.handbook == tmp_path / "challenge" / "edupulse_資料說明書.md"
    assert paths.working_csv == tmp_path / "challenge" / "working.csv"
    assert paths.train_csv == tmp_path / "challenge" / "train.csv"
    assert paths.test_csv == tmp_path / "challenge" / "test.csv"


def test_split_ready_requires_both_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    assert split_files_ready(paths) is False
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    assert split_files_ready(paths) is False
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    assert split_files_ready(paths) is True


def test_data_views_only_list_existing_files(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.start_csv.write_text("a\n1\n", encoding="utf-8")
    assert available_data_views(paths) == [DATA_VIEW_START]
    assert default_data_view(paths) == DATA_VIEW_START
    paths.working_csv.write_text("a\n1\n", encoding="utf-8")
    assert available_data_views(paths) == [DATA_VIEW_START, DATA_VIEW_WORKING]
    assert default_data_view(paths) == DATA_VIEW_WORKING
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n1\n", encoding="utf-8")
    assert available_data_views(paths) == [
        DATA_VIEW_START,
        DATA_VIEW_WORKING,
        DATA_VIEW_TRAIN,
        DATA_VIEW_TEST,
    ]
    assert default_data_view(paths) == DATA_VIEW_WORKING


def test_zones_unlock_from_split_and_artifact(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    assert model_zone_unlocked(paths) is False
    assert result_zone_unlocked(paths, artifact_present=True) is False
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n1\n", encoding="utf-8")
    assert model_zone_unlocked(paths) is True
    assert result_zone_unlocked(paths, artifact_present=False) is False
    assert result_zone_unlocked(paths, artifact_present=True) is True


def test_invalidate_split_deletes_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    invalidate_challenge_split(paths)
    assert not paths.train_csv.exists()
    assert not paths.test_csv.exists()


def test_stale_working_invalidates_split(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    paths.working_csv.write_text("a\n3\n", encoding="utf-8")
    newer = max(paths.train_csv.stat().st_mtime, paths.test_csv.stat().st_mtime) + 10
    import os

    os.utime(paths.working_csv, (newer, newer))
    assert sync_split_if_working_stale(paths) is True
    assert not paths.train_csv.exists()
    assert not paths.test_csv.exists()


def test_stale_working_invalidates_if_newer_than_either_split(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    paths.working_csv.write_text("a\n3\n", encoding="utf-8")
    import os

    train_mtime = paths.train_csv.stat().st_mtime
    os.utime(paths.working_csv, (train_mtime + 5, train_mtime + 5))
    os.utime(paths.test_csv, (train_mtime + 20, train_mtime + 20))
    assert sync_split_if_working_stale(paths) is True
    assert not paths.train_csv.exists()
    assert not paths.test_csv.exists()


def test_split_signature_changes_when_files_replaced(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    first = split_signature(paths)
    assert first is not None
    assert artifact_matches_current_split(paths, first) is True
    paths.train_csv.write_text("a\n9\n", encoding="utf-8")
    import os

    os.utime(paths.train_csv, (first[0] + 10, first[0] + 10))
    assert artifact_matches_current_split(paths, first) is False
    assert split_signature(paths) != first
    assert artifact_matches_current_split(paths, None) is False
    empty = challenge_paths(tmp_path / "other", "vitalrisk")
    assert split_signature(empty) is None


def test_clear_runtime_removes_working_and_split(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.challenge_dir.mkdir()
    paths.working_csv.write_text("a\n1\n", encoding="utf-8")
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    clear_challenge_runtime(paths)
    assert not paths.working_csv.exists()
    assert not paths.train_csv.exists()
    assert not paths.test_csv.exists()


def test_company_change_resets_when_previous_differs() -> None:
    assert company_changed_should_reset(None, "edupulse") is False
    assert company_changed_should_reset("edupulse", "edupulse") is False
    assert company_changed_should_reset("edupulse", "vitalrisk") is True


def test_restore_empty_shell_overwrites_live_ui(tmp_path: Path) -> None:
    source = tmp_path / "startup_challenge_empty_shell.py"
    dest = tmp_path / "startup_challenge_ui.py"
    source.write_text("# empty shell\n", encoding="utf-8")
    dest.write_text("# agent dirty ui\n", encoding="utf-8")
    restore_startup_challenge_ui(source, dest)
    assert dest.read_text(encoding="utf-8") == "# empty shell\n"


def test_host_context_is_workflow_not_pitch_board() -> None:
    text = challenge_host_context(
        company="edupulse",
        start_csv="dataset_streamlit_shell/workspace/challenge/edupulse.csv",
        handbook="dataset_streamlit_shell/workspace/challenge/edupulse_資料說明書.md",
        working_csv="dataset_streamlit_shell/workspace/challenge/working.csv",
        train_csv="dataset_streamlit_shell/workspace/challenge/train.csv",
        test_csv="dataset_streamlit_shell/workspace/challenge/test.csv",
        scripts_dir="dataset_streamlit_shell/scripts",
    )
    assert "【AI Startup Challenge 模式】" in text
    assert "模型區" in text
    assert "成果區" in text
    assert "Challenge 訓練資料" in text
    assert "Challenge 測試資料" in text
    assert "我們在解決什麼" not in text
    assert "白板三塊" not in text
    assert "BOARD_CUSTOMER" not in text
    assert "edupulse" in text
    assert "G1" in text or "G2" in text
    assert "nn_form.json" not in text
    assert "Challenge 工作資料" in text
    assert "只作為重置來源" not in text
    assert "不要改根目錄的 original.csv" in text
    assert "challenge_model_artifact" in text


def test_host_context_vitalrisk_fragment() -> None:
    text = challenge_host_context(
        company="vitalrisk",
        start_csv="challenge/vitalrisk.csv",
        handbook="challenge/vitalrisk_資料說明書.md",
        working_csv="challenge/working.csv",
        train_csv="challenge/train.csv",
        test_csv="challenge/test.csv",
        scripts_dir="scripts",
    )
    assert "假陽性" in text or "假陰性" in text
    assert "不是診斷" in text


def test_page_snapshot_mentions_split_not_board() -> None:
    snap = challenge_page_snapshot(
        company="edupulse",
        start_exists=True,
        working_exists=False,
        train_exists=False,
        test_exists=False,
        artifact_present=False,
    )
    assert "edupulse" in snap
    assert "Challenge 起點資料" in snap
    assert "Challenge 工作資料" in snap
    assert "Challenge 訓練資料" in snap
    assert "Challenge 測試資料" in snap
    assert "不存在" in snap
    assert "①空殼" not in snap
    assert "白板" not in snap


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
    assert (UI / "startup_challenge_empty_shell.py").is_file()


def test_empty_shell_is_workflow_page_without_pitch_chrome() -> None:
    ui = (UI / "startup_challenge_ui.py").read_text(encoding="utf-8")
    empty = (UI / "startup_challenge_empty_shell.py").read_text(encoding="utf-8")
    assert ui == empty
    assert "模型區" in ui
    assert "成果區" in ui
    assert "這裡之後放你們要展示的模型" in ui
    assert "這裡之後放訓練後的成果" in ui
    assert "BOARD_CUSTOMER" not in ui
    assert "TODO(challenge)" not in ui
    assert "上台 Gate" not in ui
    assert "建議問 Agent" not in ui
    assert "我們在解決什麼" not in ui
    assert "challenge_host_context" in ui
    assert "agent_scope" in ui or "host_context=" in ui
    greeting = (
        SHELL / "ui" / "data_ui.py"
    ).read_text(encoding="utf-8")
    assert "白板三塊" not in greeting
    assert "模型區" in greeting or "Challenge 工作資料" in greeting
