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
