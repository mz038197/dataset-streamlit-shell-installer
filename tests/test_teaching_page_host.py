from __future__ import annotations

from pathlib import Path

UI = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
    / "dataset_streamlit_shell"
    / "ui"
)


def _def_block(src: str, name: str) -> str:
    marker = f"def {name}("
    start = src.index(marker)
    lines = src[start:].splitlines()
    collected = [lines[0]]
    for line in lines[1:]:
        if line.startswith("def "):
            break
        collected.append(line)
    return "\n".join(collected)


def test_teaching_page_host_context_forbids_root_working_and_accepts_fragment() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    text = _def_block(src, "teaching_page_host_context")
    assert "內建範例資料" in text
    assert "working.csv" in text
    assert "不要讀寫 ready.csv" in text or "不要建立 ready.csv" in text
    assert "original.csv" in text
    assert "join(fragment" in text
    assert "commit_dual_table_merge" not in text
    assert "Working 工作資料路徑" not in text


TEACHING_CHAT_FILES = (
    "nn_ui.py",
    "clustering_ui.py",
    "cnn_ui.py",
    "cv_layout.py",
)


def test_teaching_pages_pass_own_host() -> None:
    for name in TEACHING_CHAT_FILES:
        src = (UI / name).read_text(encoding="utf-8")
        assert "teaching_page_host_context" in src, name
        assert "host_context=" in src, name


def test_neural_network_host_uses_form_fragment() -> None:
    src = (UI / "nn_ui.py").read_text(encoding="utf-8")
    assert "nn_host_context_fragment" in src


def test_invoke_data_agent_takes_page_host() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    text = _def_block(src, "invoke_data_agent")
    assert "host_context" in text
    assert "_remember_chat_page" in text


def test_framework_chat_has_no_data_science_default() -> None:
    data_ui = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "def dataset_base_context" not in data_ui
    assert "skip_working_snapshot" not in data_ui
    assert "lr_host_context_fragment" not in data_ui
    assert "nn_host_context_fragment" not in data_ui
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "teaching_page_host_context" not in workflow
