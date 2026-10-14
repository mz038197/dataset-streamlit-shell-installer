"""電腦視覺課程包（#15）：只裝框架 + computer-vision 即可使用六頁，host 只含一個課程包層。"""

from __future__ import annotations

import ast
from pathlib import Path

from pack_install_helpers import (
    TEMPLATE_SHELL,
    FakeDependencyRunner,
    installed_files,
    installed_shell,
    sidebar,
    third_party_imports,
    unresolved_internal_imports,
)

from add_dataset_streamlit_shell.course_packs import COMPUTER_VISION, DEEP_LEARNING, pack_files
from add_dataset_streamlit_shell.installer import FRAMEWORK_DEPENDENCIES, add_packs
from dataset_streamlit_shell.host_context import (
    assemble_host_context,
    framework_host_context,
)
from dataset_streamlit_shell.ui import computer_vision_pack

CV_PAGES = (
    "pages/27_CNN_Introduction.py",
    "pages/22_Image_Classification.py",
    "pages/23_Object_Detection.py",
    "pages/24_Semantic_Segmentation.py",
    "pages/25_Instance_Segmentation.py",
    "pages/26_Promptable_Segmentation.py",
)


def test_framework_plus_computer_vision_resolves_every_internal_import(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "computer-vision")

    files = installed_files(shell)
    assert set(CV_PAGES) <= files
    assert "built-in-data/computer-vision/examples/dog.jpg" in files
    assert "built-in-data/computer-vision/models/.gitkeep" in files
    assert not files & set(pack_files(TEMPLATE_SHELL, DEEP_LEARNING))
    assert unresolved_internal_imports(shell) == []


def test_computer_vision_does_not_share_files_with_deep_learning() -> None:
    cv = set(pack_files(TEMPLATE_SHELL, COMPUTER_VISION))
    assert not cv & set(pack_files(TEMPLATE_SHELL, DEEP_LEARNING))


def test_framework_plus_computer_vision_sidebar_shows_six_pages(tmp_path: Path) -> None:
    shell = installed_shell(tmp_path, "computer-vision")

    sections = sidebar(shell)

    assert list(sections) == ["", "電腦視覺"]
    assert len(sections["電腦視覺"]) == 6


def test_computer_vision_dependencies_match_its_third_party_imports(
    tmp_path: Path,
) -> None:
    shell = installed_shell(tmp_path, "computer-vision")
    module_to_dist = {
        "streamlit": "streamlit>=1.61",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow-cpu",
        "torch": "torch",
        "torchvision": "torchvision",
        "ultralytics": "ultralytics>=8.3.237",
        "cv2": "opencv-python-headless",
        "PIL": "pillow",
        "gdown": "gdown",
    }

    used = third_party_imports(shell, pack_files(TEMPLATE_SHELL, COMPUTER_VISION))

    assert used <= set(module_to_dist), used
    needed = {module_to_dist[name] for name in used} - set(FRAMEWORK_DEPENDENCIES)
    # SAM3SemanticPredictor 需要 ultralytics 的 CLIP（ultralytics 執行時才 import）
    clip = "clip @ git+https://github.com/ultralytics/CLIP.git"
    assert set(COMPUTER_VISION.dependencies) == needed | {clip}


def test_add_computer_vision_installs_only_its_dependencies(tmp_path: Path) -> None:
    installed_shell(tmp_path)
    runner = FakeDependencyRunner()

    add_packs(tmp_path, ["computer-vision"], dependency_runner=runner)

    assert len(runner.calls) == 1
    assert runner.added == list(COMPUTER_VISION.dependencies)
    assert {"torch", "torchvision", "ultralytics>=8.3.237", "tensorflow-cpu"} <= set(
        runner.added
    )


def test_computer_vision_host_is_framework_plus_one_pack_layer() -> None:
    layer = computer_vision_pack.computer_vision_pack_layer()
    host = computer_vision_pack.computer_vision_host_context()

    assert host == assemble_host_context(layer)
    assert host.startswith(framework_host_context())
    assert host.count(layer) == 1
    assert host.count("課程包】") == 1


def test_computer_vision_layer_uses_built_in_examples_and_writes_nothing() -> None:
    layer = computer_vision_pack.computer_vision_pack_layer()

    assert "dataset_streamlit_shell/built-in-data/computer-vision/examples" in layer
    assert "可寫清單：無" in layer
    # 白名單即可，不需點名其他課程包的檔案
    for word in ("working.csv", "original.csv", "Working", "雙表", "cleaning_log", "ready.csv"):
        assert word not in layer, word
    # 框架層已有的句子不重複
    assert "read_file" not in layer
    assert "scripts" not in layer


class _Session(dict):
    pass


def test_computer_vision_chat_passes_cv_host_and_page_state(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_render_chat_panel(extra_context="", page_name="", **kwargs) -> None:
        captured.update(kwargs, extra_context=extra_context, page_name=page_name)

    monkeypatch.setattr(computer_vision_pack, "render_chat_panel", fake_render_chat_panel)
    monkeypatch.setattr(
        computer_vision_pack.st, "session_state", _Session(cv_ctx="偵測到 3 隻貓。")
    )

    computer_vision_pack.render_computer_vision_chat(
        page_title="物件偵測", context_key="cv_ctx"
    )

    assert captured["host_context"] == computer_vision_pack.computer_vision_host_context()
    assert captured["page_name"] == "物件偵測"
    assert captured["extra_context"] == "偵測到 3 隻貓。"


def test_computer_vision_chat_defaults_page_state_to_page_title(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        computer_vision_pack,
        "render_chat_panel",
        lambda extra_context="", page_name="", **kw: captured.update(
            extra_context=extra_context
        ),
    )
    monkeypatch.setattr(computer_vision_pack.st, "session_state", _Session())

    computer_vision_pack.render_computer_vision_chat(
        page_title="語意分割", context_key="missing"
    )

    assert captured["extra_context"] == "目前頁面：語意分割。"


def _called_and_imported_names(rel: str) -> set[str]:
    tree = ast.parse((TEMPLATE_SHELL / rel).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def test_computer_vision_pages_open_chat_only_through_their_pack_layer() -> None:
    for rel in pack_files(TEMPLATE_SHELL, COMPUTER_VISION):
        if not rel.endswith(".py") or rel == "ui/computer_vision_pack.py":
            continue
        names = _called_and_imported_names(rel)
        assert "teaching_page_host_context" not in names, rel
        assert "render_chat_panel" not in names, rel
