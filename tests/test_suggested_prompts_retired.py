from __future__ import annotations

from pathlib import Path

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
SHELL = TEMPLATE_ROOT / "dataset_streamlit_shell"

HEADING = "建議問 Agent"


def test_shell_template_has_no_suggested_prompt_heading() -> None:
    hits = [
        str(path.relative_to(SHELL)).replace("\\", "/")
        for path in SHELL.rglob("*.py")
        if HEADING in path.read_text(encoding="utf-8")
    ]
    assert hits == []
