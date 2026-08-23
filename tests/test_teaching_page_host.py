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


def test_dataset_base_context_keeps_prep_rules_without_teaching_writeback() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    text = _def_block(src, "dataset_base_context")
    assert "Working 工作資料路徑" in text
    assert "Ready 分析就緒資料路徑" in text
    assert "commit_dual_table_merge" in text
    assert "【類神經網路頁】" not in text
    assert "【線性回歸頁】" not in text
    assert "nn_host_context_fragment" not in text
    assert "lr_host_context_fragment" not in text
    assert "nn_form.json" not in text
    assert "lr_slots.json" not in text


def test_teaching_page_host_context_forbids_root_working_and_accepts_fragment() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    text = _def_block(src, "teaching_page_host_context")
    assert "內建範例資料" in text
    assert "working.csv" in text
    assert "ready.csv" in text
    assert "original.csv" in text
    assert "join(fragment" in text
    assert "commit_dual_table_merge" not in text
    assert "Working 工作資料路徑" not in text


TEACHING_CHAT_FILES = (
    "lr_ui.py",
    "nn_ui.py",
    "logistic_regression_ui.py",
    "svm_ui.py",
    "knn_ui.py",
    "tree_ui.py",
    "clustering_ui.py",
    "cnn_ui.py",
    "cv_layout.py",
)


def test_teaching_pages_pass_own_host_and_skip_working_snapshot() -> None:
    for name in TEACHING_CHAT_FILES:
        src = (UI / name).read_text(encoding="utf-8")
        assert "teaching_page_host_context" in src, name
        assert "skip_working_snapshot=True" in src, name
        assert "host_context=" in src, name


def test_linear_regression_host_uses_slot_fragment() -> None:
    src = (UI / "lr_ui.py").read_text(encoding="utf-8")
    assert "lr_host_context_fragment" in src
    assert "skip_working_snapshot=True" in src


def test_neural_network_host_uses_form_fragment() -> None:
    src = (UI / "nn_ui.py").read_text(encoding="utf-8")
    assert "nn_host_context_fragment" in src


def test_invoke_data_agent_takes_page_host() -> None:
    src = (UI / "data_ui.py").read_text(encoding="utf-8")
    text = _def_block(src, "invoke_data_agent")
    assert "host_context" in text
    assert "_remember_chat_page" in text


def test_prep_pages_keep_dataset_base_default() -> None:
    data_ui = (UI / "data_ui.py").read_text(encoding="utf-8")
    assert "def dataset_base_context" in data_ui
    assert "lr_host_context_fragment" not in data_ui
    assert "nn_host_context_fragment" not in data_ui
    workflow = (UI / "workflow_ui.py").read_text(encoding="utf-8")
    assert "teaching_page_host_context" not in workflow
    assert "skip_working_snapshot=True" not in workflow
