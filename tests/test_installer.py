from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from add_dataset_streamlit_shell import installer
from add_dataset_streamlit_shell.installer import (
    CHARSET_NORMALIZER_CONSTRAINT,
    PROJECT_DEPENDENCIES,
    install_shell,
)


def test_install_shell_copies_template_without_agent_core(tmp_path: Path) -> None:
    result = install_shell(tmp_path, install_dependencies=False)

    assert (result.target / "app.py").exists()
    assert (result.target / "ui" / "data_ui.py").exists()
    assert (result.target / "pages" / "1_Database.py").exists()
    assert (result.target / "workspace" / ".gitkeep").exists()
    assert (tmp_path / "sessions" / ".gitkeep").exists()
    assert not (result.target / "sessions").exists()
    data_ui = (result.target / "ui" / "data_ui.py").read_text(encoding="utf-8")
    assert 'SESSION_DIR = PROJECT_ROOT / "sessions"' in data_ui


def test_install_shell_can_require_agent_core(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="agent_core.py"):
        install_shell(tmp_path, require_agent_core=True, install_dependencies=False)


def test_install_shell_refuses_existing_target(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    install_shell(tmp_path, install_dependencies=False)

    with pytest.raises(FileExistsError, match="already exists"):
        install_shell(tmp_path, install_dependencies=False)


def test_install_shell_force_backs_up_existing_target(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    install_shell(tmp_path, install_dependencies=False)

    result = install_shell(tmp_path, force=True, install_dependencies=False)

    assert result.backed_up_to is not None
    assert result.backed_up_to.exists()
    assert result.target.exists()


def test_install_shell_update_preserves_runtime_data(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")
    result = install_shell(tmp_path, install_dependencies=False)
    target = result.target
    original_app = (target / "app.py").read_text(encoding="utf-8")

    (target / "app.py").write_text("# stale app\n", encoding="utf-8")
    (target / "old_page.py").write_text("# stale page\n", encoding="utf-8")
    (target / "workspace" / "original.csv").write_text("a\n1\n", encoding="utf-8")
    (target / "workspace" / "working.csv").write_text("a\n2\n", encoding="utf-8")
    (target / "workspace" / "ready.csv").write_text("a\n3\n", encoding="utf-8")
    (target / "workspace" / "cleaning_log.jsonl").write_text("{}\n", encoding="utf-8")
    # 新版：專案根 sessions/
    root_sessions = tmp_path / "sessions"
    root_sessions.mkdir(exist_ok=True)
    (root_sessions / "session_20260529_000000_root.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    # 舊版：shell 內 sessions/（--update 仍須保留、不可刪）
    legacy_sessions = target / "sessions"
    legacy_sessions.mkdir(exist_ok=True)
    (legacy_sessions / "session_20260529_000000_legacy.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    memory_dir = target / "memory"
    memory_dir.mkdir(exist_ok=True)
    (memory_dir / "notes.jsonl").write_text('{"keep":true}\n', encoding="utf-8")
    scripts_dir = target / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    (scripts_dir / "student_cleanup.py").write_text("# keep me\n", encoding="utf-8")
    upload_dir = target / "uploads" / "chat_images"
    upload_dir.mkdir(parents=True)
    (upload_dir / "image.png").write_bytes(b"png")

    updated = install_shell(tmp_path, update=True, install_dependencies=False)

    assert updated.target == target
    assert updated.backed_up_to is None
    assert (target / "app.py").read_text(encoding="utf-8") == original_app
    assert not (target / "old_page.py").exists()
    assert (target / "workspace" / "original.csv").read_text(encoding="utf-8") == "a\n1\n"
    assert (target / "workspace" / "working.csv").read_text(encoding="utf-8") == "a\n2\n"
    assert (target / "workspace" / "ready.csv").read_text(encoding="utf-8") == "a\n3\n"
    assert (target / "workspace" / "cleaning_log.jsonl").read_text(encoding="utf-8") == "{}\n"
    assert (root_sessions / "session_20260529_000000_root.jsonl").exists()
    assert (legacy_sessions / "session_20260529_000000_legacy.jsonl").exists()
    assert (tmp_path / "sessions" / ".gitkeep").exists()
    assert (memory_dir / "notes.jsonl").read_text(encoding="utf-8") == '{"keep":true}\n'
    assert (scripts_dir / "student_cleanup.py").read_text(encoding="utf-8") == "# keep me\n"
    assert (upload_dir / "image.png").read_bytes() == b"png"


def test_install_shell_update_preserves_sam3_weights(tmp_path: Path) -> None:
    install_shell(tmp_path, install_dependencies=False)
    target = tmp_path / "dataset_streamlit_shell"
    models_dir = target / "built-in-data" / "computer-vision" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = b"PKfake-sam3-weights"
    (models_dir / "sam3.pt").write_bytes(payload)
    (target / "app.py").write_text("# stale app\n", encoding="utf-8")

    install_shell(tmp_path, update=True, install_dependencies=False)

    assert (models_dir / "sam3.pt").read_bytes() == payload


def test_install_shell_update_installs_when_target_missing(tmp_path: Path) -> None:
    (tmp_path / "agent_core.py").write_text("class Agent: ...\n", encoding="utf-8")

    result = install_shell(tmp_path, update=True, install_dependencies=False)

    assert (result.target / "app.py").exists()


def test_install_shell_installs_project_dependencies_by_default(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_runner(args: list[str], *, cwd: Path, check: bool) -> None:
        calls.append((args, cwd, check))

    result = install_shell(tmp_path, dependency_runner=fake_runner)

    assert len(calls) == 3
    assert calls[0] == (
        ["uv", "add", "--upgrade-package", "openai-tts", *PROJECT_DEPENDENCIES],
        tmp_path.resolve(),
        True,
    )
    assert calls[1][0][:2] == ["uv", "add"]
    assert "peas-agent-runtime" in " ".join(str(x) for x in calls[1][0])
    assert calls[1][1:] == (tmp_path.resolve(), True)
    assert calls[2] == (
        ["uv", "add", CHARSET_NORMALIZER_CONSTRAINT],
        tmp_path.resolve(),
        True,
    )
    assert "streamlit>=1.50" in PROJECT_DEPENDENCIES
    assert PROJECT_DEPENDENCIES == result.installed_dependencies[:-2]
    assert "peas-agent-runtime" in result.installed_dependencies[-2]
    assert result.installed_dependencies[-1] == CHARSET_NORMALIZER_CONSTRAINT


def test_check_contract_uses_uv_run_not_global_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "main_shell.py").write_text(
        "def create_agent(session_path=None, host_context=None):\n"
        "    class A:\n"
        "        def chat(self, text, *, image_path=None, on_token=None):\n"
        "            return text\n"
        "    return A()\n",
        encoding="utf-8",
    )
    uv_calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        uv_calls.append(list(args))
        assert args[:3] == ["uv", "run", "python"]
        assert cwd == tmp_path.resolve()
        assert args[-1] == "main_shell:create_agent"
        payload = json.dumps(
            {
                "skip": False,
                "ok": True,
                "messages": ["提示：測試略過深檢"],
                "factory_ref": "main_shell:create_agent",
            }
        )
        return SimpleNamespace(returncode=0, stdout=payload + "\n", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", fake_run)

    result = install_shell(tmp_path, install_dependencies=False)

    assert result.contract_ok is True
    assert result.factory_ref == "main_shell:create_agent"
    assert any("契約通過：main_shell:create_agent" in m for m in result.contract_messages)
    assert uv_calls
    assert "peas-agent-runtime" not in " ".join(uv_calls[0][:3])
