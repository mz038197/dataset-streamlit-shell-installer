"""CLI 合約（#11）：不帶子指令裝框架、add、list。"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from add_dataset_streamlit_shell.cli import app
from add_dataset_streamlit_shell.course_packs import COURSE_PACKS

runner = CliRunner()
SHELL = "dataset_streamlit_shell"


def _invoke(*args: str):
    return runner.invoke(app, list(args))


def _record(project: Path) -> list[str]:
    data = json.loads((project / SHELL / "course_packs.json").read_text(encoding="utf-8"))
    return data["course_packs"]


def test_bare_command_installs_framework_and_suggests_adding_a_pack(tmp_path: Path) -> None:
    result = _invoke("-C", str(tmp_path), "--no-install-deps")

    assert result.exit_code == 0, result.output
    assert (tmp_path / SHELL / "app.py").is_file()
    assert not (tmp_path / SHELL / "pages").exists()
    assert _record(tmp_path) == []
    assert "add data-science" in result.output


def test_add_installs_named_packs_into_project_root(tmp_path: Path) -> None:
    result = _invoke("add", "data-science", "-C", str(tmp_path), "--no-install-deps")

    assert result.exit_code == 0, result.output
    assert (tmp_path / SHELL / "pages" / "3_Field_Quality.py").is_file()
    assert _record(tmp_path) == ["data-science"]
    assert "data-science" in result.output


def test_add_again_reports_already_installed(tmp_path: Path) -> None:
    _invoke("add", "data-science", "-C", str(tmp_path), "--no-install-deps")

    result = _invoke("add", "data-science", "-C", str(tmp_path), "--no-install-deps")

    assert result.exit_code == 0, result.output
    assert "已安裝" in result.output


def test_add_unknown_name_fails_with_available_names(tmp_path: Path) -> None:
    result = _invoke("add", "nope", "-C", str(tmp_path), "--no-install-deps")

    assert result.exit_code == 1
    assert "nope" in result.output
    for pack in COURSE_PACKS:
        assert pack.name in result.output
    assert not (tmp_path / SHELL).exists()


def test_list_shows_packs_sections_and_installed_state(tmp_path: Path) -> None:
    _invoke("add", "data-science", "-C", str(tmp_path), "--no-install-deps")

    result = _invoke("list", "-C", str(tmp_path))

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    for pack in COURSE_PACKS:
        line = next(line for line in lines if pack.name in line)
        assert pack.sidebar_section in line
        assert ("已安裝" in line) == (pack.name == "data-science"), line


def test_update_without_record_fails_and_suggests_force(tmp_path: Path) -> None:
    _invoke("-C", str(tmp_path), "--no-install-deps")
    (tmp_path / SHELL / "course_packs.json").unlink()

    result = _invoke("-C", str(tmp_path), "--update", "--no-install-deps")

    assert result.exit_code == 1
    assert "--force" in result.output
