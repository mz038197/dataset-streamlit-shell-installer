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
    challenge_agent_scope,
    clear_challenge_runtime,
    company_changed_should_reset,
    default_data_view,
    apply_confirmed_company,
    artifact_path,
    invalidate_challenge_artifact,
    legacy_shared_runtime_exists,
    load_company_ui,
    migrate_legacy_runtime,
    read_committed_company,
    resolve_startup_company,
    save_ui_snapshot,
    ui_snapshot_path,
    write_committed_company,
    artifact_matches_current_split,
    invalidate_challenge_split,
    model_zone_unlocked,
    resolve_company_switch,
    COMPANY_SWITCH_CANCEL_LABEL,
    COMPANY_SWITCH_CONFIRM_LABEL,
    COMPANY_SWITCH_DIALOG_TITLE,
    company_switch_dialog_body,
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


def test_all_companies_ship_start_csv_and_handbook() -> None:
    challenge = SHELL / "workspace" / "challenge"
    for company in CHALLENGE_COMPANIES:
        csv_path = challenge / f"{company}.csv"
        handbook = challenge / f"{company}_資料說明書.md"
        assert csv_path.is_file(), csv_path.name
        assert csv_path.stat().st_size > 0
        assert handbook.is_file(), handbook.name
        text = handbook.read_text(encoding="utf-8")
        assert company in text.lower()
    assert list(challenge.glob("*教師用*")) == []
    assert list(challenge.glob("**/Raw/**")) == []


def test_challenge_paths_include_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    assert paths.start_csv == tmp_path / "challenge" / "edupulse.csv"
    assert paths.handbook == tmp_path / "challenge" / "edupulse_資料說明書.md"
    assert paths.working_csv == tmp_path / "challenge" / "edupulse" / "working.csv"
    assert paths.train_csv == tmp_path / "challenge" / "edupulse" / "train.csv"
    assert paths.test_csv == tmp_path / "challenge" / "edupulse" / "test.csv"


def test_split_ready_requires_both_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.working_csv.parent.mkdir(parents=True)
    assert split_files_ready(paths) is False
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    assert split_files_ready(paths) is False
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    assert split_files_ready(paths) is True


def test_data_views_only_list_existing_files(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.working_csv.parent.mkdir(parents=True)
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
    paths.working_csv.parent.mkdir(parents=True)
    assert model_zone_unlocked(paths) is False
    assert result_zone_unlocked(paths, artifact_present=True) is False
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n1\n", encoding="utf-8")
    assert model_zone_unlocked(paths) is True
    assert result_zone_unlocked(paths, artifact_present=False) is False
    assert result_zone_unlocked(paths, artifact_present=True) is True


def test_invalidate_split_deletes_train_and_test(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.working_csv.parent.mkdir(parents=True)
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    invalidate_challenge_split(paths)
    assert not paths.train_csv.exists()
    assert not paths.test_csv.exists()


def test_stale_working_invalidates_split(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.working_csv.parent.mkdir(parents=True)
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
    paths.working_csv.parent.mkdir(parents=True)
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
    paths.working_csv.parent.mkdir(parents=True)
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
    paths.working_csv.parent.mkdir(parents=True)
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


def test_resolve_company_switch_prompts_until_confirmed() -> None:
    assert resolve_company_switch(None, "edupulse") == ("edupulse", None)
    assert resolve_company_switch(
        None, "edupulse", require_first_confirm=True
    ) == (None, "edupulse")
    assert resolve_company_switch("edupulse", "edupulse") == ("edupulse", None)
    assert resolve_company_switch("edupulse", "vitalrisk") == ("edupulse", "vitalrisk")
    assert resolve_company_switch("edupulse", "edupulse", "vitalrisk") == (
        "edupulse",
        "vitalrisk",
    )
    assert resolve_company_switch("edupulse", "airsense", "vitalrisk") == (
        "edupulse",
        "airsense",
    )


def test_company_switch_confirm_copy() -> None:
    assert COMPANY_SWITCH_DIALOG_TITLE == "確定更換挑戰公司？"
    assert COMPANY_SWITCH_CONFIRM_LABEL == "確認更換"
    assert COMPANY_SWITCH_CANCEL_LABEL == "取消"
    assert company_switch_dialog_body("vitalrisk") == (
        "切換後會改看該公司的資料、畫面與對話，不會刪檔。再切回來進度還在。"
        "確定改為 **vitalrisk**？"
    )


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
        working_csv="dataset_streamlit_shell/workspace/challenge/edupulse/working.csv",
        train_csv="dataset_streamlit_shell/workspace/challenge/edupulse/train.csv",
        test_csv="dataset_streamlit_shell/workspace/challenge/edupulse/test.csv",
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
        working_csv="challenge/vitalrisk/working.csv",
        train_csv="challenge/vitalrisk/train.csv",
        test_csv="challenge/vitalrisk/test.csv",
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
    assert "st.dialog" in ui
    assert "resolve_company_switch" in ui
    assert "COMPANY_SWITCH_DIALOG_TITLE" in ui
    assert "apply_confirmed_company" in ui
    assert "challenge_agent_scope" in ui
    assert "clear_challenge_runtime" not in ui
    greeting = (
        SHELL / "ui" / "data_ui.py"
    ).read_text(encoding="utf-8")
    assert "白板三塊" not in greeting
    assert "模型區" in greeting or "Challenge 工作資料" in greeting
    assert 'agent_scope.startswith("challenge")' in greeting


def test_legacy_files_ignore_session_until_confirmed(tmp_path: Path) -> None:
    challenge_dir = tmp_path / "challenge"
    challenge_dir.mkdir()
    (challenge_dir / "working.csv").write_text("old\n", encoding="utf-8")
    company, require_first = resolve_startup_company(
        session_company="edupulse",
        challenge_dir=challenge_dir,
    )
    assert require_first is True
    assert company is None
    assert (challenge_dir / "working.csv").read_text(encoding="utf-8") == "old\n"

    write_committed_company(challenge_dir, "vitalrisk")
    company, require_first = resolve_startup_company(
        session_company="edupulse",
        challenge_dir=challenge_dir,
    )
    assert require_first is False
    assert company == "vitalrisk"


def test_committed_company_roundtrip(tmp_path: Path) -> None:
    challenge_dir = tmp_path / "challenge"
    assert read_committed_company(challenge_dir) is None
    write_committed_company(challenge_dir, "vitalrisk")
    assert read_committed_company(challenge_dir) == "vitalrisk"
    (challenge_dir / "committed_company").write_text("not-a-company\n", encoding="utf-8")
    assert read_committed_company(challenge_dir) is None


def test_legacy_shared_runtime_and_migrate_into_confirmed_company(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "challenge"
    challenge_dir.mkdir()
    (challenge_dir / "working.csv").write_text("w\n1\n", encoding="utf-8")
    (challenge_dir / "train.csv").write_text("t\n1\n", encoding="utf-8")
    (challenge_dir / "test.csv").write_text("e\n1\n", encoding="utf-8")
    assert legacy_shared_runtime_exists(challenge_dir) is True
    assert migrate_legacy_runtime(challenge_dir, "vitalrisk") is True
    dest = challenge_dir / "vitalrisk"
    assert (dest / "working.csv").read_text(encoding="utf-8") == "w\n1\n"
    assert (dest / "train.csv").read_text(encoding="utf-8") == "t\n1\n"
    assert (dest / "test.csv").read_text(encoding="utf-8") == "e\n1\n"
    assert not (challenge_dir / "working.csv").exists()
    assert not (challenge_dir / "train.csv").exists()
    assert not (challenge_dir / "test.csv").exists()
    assert legacy_shared_runtime_exists(challenge_dir) is False


def test_migrate_legacy_does_not_overwrite_existing_company_working(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "challenge"
    dest = challenge_dir / "edupulse"
    dest.mkdir(parents=True)
    (dest / "working.csv").write_text("keep\n", encoding="utf-8")
    (challenge_dir / "working.csv").write_text("old\n", encoding="utf-8")
    assert migrate_legacy_runtime(challenge_dir, "edupulse") is False
    assert (dest / "working.csv").read_text(encoding="utf-8") == "keep\n"
    assert (challenge_dir / "working.csv").read_text(encoding="utf-8") == "old\n"


def test_apply_confirmed_company_keeps_working_and_swaps_ui_snapshot(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    live = tmp_path / "startup_challenge_ui.py"
    empty = tmp_path / "startup_challenge_empty_shell.py"
    empty.write_text("# empty\n", encoding="utf-8")
    live.write_text("# vitalrisk ui\n", encoding="utf-8")
    vital = challenge_paths(workspace, "vitalrisk")
    vital.working_csv.parent.mkdir(parents=True)
    vital.working_csv.write_text("cleaned\n", encoding="utf-8")
    vital.train_csv.write_text("tr\n", encoding="utf-8")
    vital.test_csv.write_text("te\n", encoding="utf-8")

    apply_confirmed_company(
        previous="vitalrisk",
        company="airsense",
        workspace_dir=workspace,
        live_ui=live,
        empty_shell=empty,
    )
    assert vital.working_csv.read_text(encoding="utf-8") == "cleaned\n"
    assert vital.train_csv.read_text(encoding="utf-8") == "tr\n"
    assert vital.test_csv.read_text(encoding="utf-8") == "te\n"
    assert ui_snapshot_path(vital).read_text(encoding="utf-8") == "# vitalrisk ui\n"
    assert live.read_text(encoding="utf-8") == "# empty\n"
    assert read_committed_company(workspace / "challenge") == "airsense"

    apply_confirmed_company(
        previous="airsense",
        company="vitalrisk",
        workspace_dir=workspace,
        live_ui=live,
        empty_shell=empty,
    )
    assert live.read_text(encoding="utf-8") == "# vitalrisk ui\n"


def test_invalidate_split_also_removes_artifact(tmp_path: Path) -> None:
    paths = challenge_paths(tmp_path, "edupulse")
    paths.working_csv.parent.mkdir(parents=True)
    paths.train_csv.write_text("a\n1\n", encoding="utf-8")
    paths.test_csv.write_text("a\n2\n", encoding="utf-8")
    artifact_path(paths).write_text("{}\n", encoding="utf-8")
    invalidate_challenge_split(paths)
    invalidate_challenge_artifact(paths)
    assert not paths.train_csv.exists()
    assert not artifact_path(paths).exists()


def test_challenge_agent_scope_is_per_company() -> None:
    assert challenge_agent_scope("edupulse") == "challenge_edupulse"
    assert challenge_agent_scope("vitalrisk") != challenge_agent_scope("edupulse")


def test_save_and_load_ui_snapshot(tmp_path: Path) -> None:
    live = tmp_path / "live.py"
    snap = tmp_path / "company" / "startup_challenge_ui.py"
    empty = tmp_path / "empty.py"
    live.write_text("# live\n", encoding="utf-8")
    empty.write_text("# empty\n", encoding="utf-8")
    save_ui_snapshot(live, snap)
    live.write_text("# dirty\n", encoding="utf-8")
    load_company_ui(live, snap, empty)
    assert live.read_text(encoding="utf-8") == "# live\n"
    missing = tmp_path / "missing.py"
    load_company_ui(live, missing, empty)
    assert live.read_text(encoding="utf-8") == "# empty\n"
