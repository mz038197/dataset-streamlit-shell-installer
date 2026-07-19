from __future__ import annotations

import sys
from pathlib import Path

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
)
sys.path.insert(0, str(TEMPLATE))

from agent_loader import load_create_agent  # noqa: E402


def test_load_main_shell_create_agent(tmp_path: Path) -> None:
    (tmp_path / "main_shell.py").write_text(
        """
def create_agent(session_path=None, host_context=None):
    return {"session_path": session_path, "host_context": host_context}
""",
        encoding="utf-8",
    )
    result = load_create_agent(tmp_path)
    assert result.factory is not None
    assert result.factory_ref == "main_shell:create_agent"
    agent = result.factory(session_path="s.jsonl", host_context="HOST")
    assert agent["host_context"] == "HOST"


def test_load_falls_back_to_main(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text(
        """
def create_agent(session_path=None, host_context=None):
    return "ok"
""",
        encoding="utf-8",
    )
    result = load_create_agent(tmp_path)
    assert result.factory_ref == "main:create_agent"


def test_no_factory_mentions_create_agent(tmp_path: Path) -> None:
    result = load_create_agent(tmp_path)
    assert result.factory is None
    assert result.error is not None
    assert "create_agent" in result.error
