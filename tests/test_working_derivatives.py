from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TEMPLATE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "add_dataset_streamlit_shell"
    / "templates"
)
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from dataset_streamlit_shell.ui import data_ui  # noqa: E402


def _patch_workspace(monkeypatch, tmp_path: Path) -> dict[str, Path]:
    paths = {
        "working": tmp_path / "working.csv",
        "ready": tmp_path / "ready.csv",
        "train": tmp_path / "train.csv",
        "val": tmp_path / "val.csv",
        "test": tmp_path / "test.csv",
    }
    monkeypatch.setattr(data_ui, "WORKING_DATASET_PATH", paths["working"])
    monkeypatch.setattr(data_ui, "READY_DATASET_PATH", paths["ready"])
    monkeypatch.setattr(data_ui, "TRAIN_DATASET_PATH", paths["train"])
    monkeypatch.setattr(data_ui, "VAL_DATASET_PATH", paths["val"])
    monkeypatch.setattr(data_ui, "TEST_DATASET_PATH", paths["test"])
    monkeypatch.setattr(
        data_ui,
        "SPLIT_DATASET_PATHS",
        (paths["train"], paths["val"], paths["test"]),
    )
    monkeypatch.setattr(data_ui, "_ensure_workspace_dir", lambda: None)
    monkeypatch.setattr(data_ui, "refresh_ready_dataset_cache", lambda: None)
    return paths


def test_saving_working_clears_splits_and_leftover_ready(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_workspace(monkeypatch, tmp_path)
    paths["ready"].write_text("a\n1\n", encoding="utf-8")
    paths["train"].write_text("a\n1\n", encoding="utf-8")
    paths["val"].write_text("a\n1\n", encoding="utf-8")
    paths["test"].write_text("a\n1\n", encoding="utf-8")

    data_ui.save_dataset(pd.DataFrame({"a": [2]}), working=True)

    assert paths["working"].is_file()
    assert not paths["ready"].exists()
    assert not paths["train"].exists()
    assert not paths["val"].exists()
    assert not paths["test"].exists()


def test_stale_working_invalidates_split_files(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_workspace(monkeypatch, tmp_path)
    paths["train"].write_text("a\n1\n", encoding="utf-8")
    paths["val"].write_text("a\n1\n", encoding="utf-8")
    paths["test"].write_text("a\n1\n", encoding="utf-8")
    paths["working"].write_text("a\n2\n", encoding="utf-8")
    newer = max(path.stat().st_mtime for path in (paths["train"], paths["val"], paths["test"])) + 10
    import os

    os.utime(paths["working"], (newer, newer))

    assert data_ui.sync_split_if_working_stale() is True
    assert not paths["train"].exists()
    assert not paths["val"].exists()
    assert not paths["test"].exists()


def test_clear_to_dual_start_deletes_leftover_ready(tmp_path: Path, monkeypatch) -> None:
    paths = _patch_workspace(monkeypatch, tmp_path)
    monkeypatch.setattr(data_ui, "WORKSPACE_DIR", tmp_path)
    monkeypatch.setattr(data_ui, "ORIGINAL_DATASET_PATH", tmp_path / "original.csv")
    monkeypatch.setattr(data_ui, "refresh_working_dataset_cache", lambda: None)
    paths["ready"].write_text("a\n1\n", encoding="utf-8")
    paths["working"].write_text("a\n2\n", encoding="utf-8")
    (tmp_path / "original.csv").write_text("a\n0\n", encoding="utf-8")

    class _Session(dict):
        def pop(self, key, default=None):
            return dict.pop(self, key, default)

    monkeypatch.setattr(data_ui.st, "session_state", _Session())
    data_ui.clear_to_dual_start()
    assert not paths["ready"].exists()
    assert not paths["working"].exists()
