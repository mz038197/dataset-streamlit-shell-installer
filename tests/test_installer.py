from __future__ import annotations

from pathlib import Path

import pytest

from add_dataset_streamlit_shell.installer import install_shell


def test_install_shell_copies_template(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")

    result = install_shell(tmp_path)

    assert (result.target / "app.py").exists()
    assert (result.target / "data_ui.py").exists()
    assert (result.target / "pages" / "1_Database.py").exists()
    assert (result.target / "data" / ".gitkeep").exists()
    assert (result.target / "sessions" / ".gitkeep").exists()


def test_install_shell_requires_agent_core(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="agent_core.py"):
        install_shell(tmp_path)


def test_install_shell_refuses_existing_target(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    install_shell(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        install_shell(tmp_path)


def test_install_shell_force_backs_up_existing_target(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    install_shell(tmp_path)

    result = install_shell(tmp_path, force=True)

    assert result.backed_up_to is not None
    assert result.backed_up_to.exists()
    assert result.target.exists()


def test_install_shell_update_preserves_runtime_data(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    result = install_shell(tmp_path)
    target = result.target
    original_app = (target / "app.py").read_text(encoding="utf-8")

    (target / "app.py").write_text("# stale app\n", encoding="utf-8")
    (target / "old_page.py").write_text("# stale page\n", encoding="utf-8")
    (target / "data" / "original.csv").write_text("a\n1\n", encoding="utf-8")
    (target / "data" / "working.csv").write_text("a\n2\n", encoding="utf-8")
    (target / "data" / "ready.csv").write_text("a\n3\n", encoding="utf-8")
    (target / "data" / "cleaning_log.jsonl").write_text("{}\n", encoding="utf-8")
    (target / "sessions" / "session_20260529_000000_abc123.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    scripts_dir = target / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "student_cleanup.py").write_text("# keep me\n", encoding="utf-8")
    upload_dir = target / "uploads" / "chat_images"
    upload_dir.mkdir(parents=True)
    (upload_dir / "image.png").write_bytes(b"png")

    updated = install_shell(tmp_path, update=True)

    assert updated.target == target
    assert updated.backed_up_to is None
    assert (target / "app.py").read_text(encoding="utf-8") == original_app
    assert not (target / "old_page.py").exists()
    assert (target / "data" / "original.csv").read_text(encoding="utf-8") == "a\n1\n"
    assert (target / "data" / "working.csv").read_text(encoding="utf-8") == "a\n2\n"
    assert (target / "data" / "ready.csv").read_text(encoding="utf-8") == "a\n3\n"
    assert (target / "data" / "cleaning_log.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert (target / "sessions" / "session_20260529_000000_abc123.jsonl").exists()
    assert (scripts_dir / "student_cleanup.py").read_text(encoding="utf-8") == "# keep me\n"
    assert (upload_dir / "image.png").read_bytes() == b"png"


def test_install_shell_update_installs_when_target_missing(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")

    result = install_shell(tmp_path, update=True)

    assert (result.target / "app.py").exists()
